import math
from math import pi
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.adaln import AdaLNSelfAttn
from models.cond_mlp import SimpleMLPAdaLN
from diffusion import create_diffusion


class ARHead_diff(nn.Module):
    def __init__(self, token_embed_dim, decoder_embed_dim, inner_ar_width=768, 
                 inner_ar_depth=1, num_sampling_steps='100', head_width=1024, head_depth=6):
        super(ARHead_diff, self).__init__()
        self.token_embed_dim = token_embed_dim
        self.width = inner_ar_width
        
        # Input projection
        self.input_proj = nn.Linear(1, inner_ar_width)
        self.cond_proj = nn.Linear(decoder_embed_dim, inner_ar_width)

        # Start token and position embedding
        self.start_token = nn.Parameter(torch.empty(1, 1, inner_ar_width))
        self.pos_embedding = nn.Parameter(torch.empty(1, token_embed_dim, inner_ar_width))

        # Backbone blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.drop_path_rate = 0.
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, inner_ar_depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=decoder_embed_dim,
                block_idx=block_idx, embed_dim=inner_ar_width, norm_layer=norm_layer, num_heads=16, mlp_ratio=4.,
                drop=0., attn_drop=0., drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=False, shared_aln=False,
                flash_if_available=True, fused_if_available=True,
            )
            for block_idx in range(inner_ar_depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        self.init_weights()

        self.net = SimpleMLPAdaLN(
            in_channels=1,  # feature-by-feature diffusion
            model_channels=head_width,
            out_channels=1 * 2,  # for vlb loss
            z_channels=inner_ar_width,
            num_res_blocks=head_depth,  # hacking
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

 
    def forward(self, z, target, mask=None):
        bsz = z.shape[0]
        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        x = torch.cat((start, self.input_proj(target[:, :-1].unsqueeze(-1))), dim=1)
        x = x + self.pos_embedding.expand(bsz, -1, -1)

        for b in self.blocks:
            x = b(x=x, cond_BD=z, attn_bias=None, causal=True)
        
        x = x.reshape(-1, self.width)
        target = target.reshape(-1, 1)

        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=x)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"].reshape(bsz, self.token_embed_dim).mean(-1)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0, top_p=0.99):
        bsz = z.shape[0]

        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        for b in self.blocks: b.attn.kv_caching(True)
        x = start
        res = []

        for i in range(self.token_embed_dim):
            x = x + self.pos_embedding[:, i:i+1].expand(bsz, 1, -1)
            for b in self.blocks:
                x = b(x=x, cond_BD=z, attn_bias=None, causal=False)

            if not cfg == 1.0:
                noise = torch.randn(x.shape[0] // 2, 1).cuda()
                noise = torch.cat([noise, noise], dim=0)
                model_kwargs = dict(c=x, cfg_scale=cfg)
                sample_fn = self.net.forward_with_cfg
            else:
                noise = torch.randn(bsz, 1).cuda()
                model_kwargs = dict(c=x.squeeze(1))
                sample_fn = self.net.forward

            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature
            )

            res.append(sampled_token_latent)

            x = self.input_proj(sampled_token_latent.unsqueeze(-1))
        
        for b in self.blocks: b.attn.kv_caching(False)
        res = torch.cat(res, dim=1)

        return res

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_std=0.02, conv_std_or_gain=0.02):
        nn.init.trunc_normal_(self.start_token.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pos_embedding.data, mean=0, std=init_std)

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.width:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.width].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
   
