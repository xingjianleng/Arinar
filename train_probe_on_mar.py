import os
import math
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

from util.loader import CachedFolder
from models import mar
from models.vae import DiagonalGaussianDistribution, AutoencoderKL
from models.uncond_mlp import UncondSimpleMLP
from models.mar import mask_by_order


def get_args_parser():
    parser = argparse.ArgumentParser('Train a probe on MAR', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')
    parser.add_argument('--norm_scale', default=0.2325, type=float,
                        help='normalization scale for vae latents')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    ## MAR params
    parser.add_argument('--enc_dec_depth', type=int, default=-1,
                        help='Encoder/Decoder depth')
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    # Second layer AR parameters
    parser.add_argument('--head_type', type=str, 
                        # choices=['ar_gmm', 'ar_diff_loss', 'gmm_wo_ar', 'gmm_cov_wo_ar', 'ar_byte'], 
                        default='ar_gmm', help='head type (default: ar_gmm)')
    parser.add_argument('--num_gaussians', type=int, default=1)
    parser.add_argument('--inner_ar_width', type=int, default=1024)
    parser.add_argument('--inner_ar_depth', type=int, default=1)
    parser.add_argument('--head_width', type=int, default=1024)
    parser.add_argument('--head_depth', type=int, default=6)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--feature_group', type=int, default=1)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--head_batch_mul', type=int, default=1)

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--evaluate', action='store_true')

    return parser

parser = get_args_parser()
args = parser.parse_args()

def save_images(tokens, path):
    os.makedirs(path, exist_ok=True)
    # unpatchify
    sampled_tokens = model.unpatchify(tokens)

    sampled_images = vae.decode(sampled_tokens / 0.2325)
    sampled_images = sampled_images.detach().cpu()
    sampled_images = (sampled_images + 1) / 2

    for b_id in range(sampled_images.size(0)):
        gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(path, '{}.png'.format(str(b_id).zfill(5))), gen_img)


dataset_train = CachedFolder(args.cached_path)
data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

kwargs = {
        "num_sampling_steps": args.num_sampling_steps,
        "enc_dec_depth": args.enc_dec_depth,
    }
model = mar.__dict__[args.model](
            img_size=args.img_size,
            vae_stride=args.vae_stride,
            patch_size=args.patch_size,
            vae_embed_dim=args.vae_embed_dim,
            mask_ratio_min=args.mask_ratio_min,
            label_drop_prob=args.label_drop_prob,
            class_num=args.class_num,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            buffer_size=args.buffer_size,
            num_gaussians=args.num_gaussians,
            grad_checkpointing=args.grad_checkpointing,
            inner_ar_width=args.inner_ar_width,
            inner_ar_depth=args.inner_ar_depth,
            head_width=args.head_width,
            head_depth=args.head_depth,
            head_type=args.head_type,
            **kwargs
        )

print("Model = %s" % str(model))
# following timm: set wd as 0 for bias and norm layers
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {}M".format(n_params / 1e6))

model.to(args.device)
checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
print("Model loaded from %s" % args.resume)

vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
for param in vae.parameters():
    param.requires_grad = False

probe = UncondSimpleMLP(
    in_channels=768,
    model_channels=1024,
    out_channels=args.vae_embed_dim,
    num_res_blocks=6,
    grad_checkpointing=False
)
probe.to(args.device)

##TODO
if not args.evaluate:
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    for step, (samples, labels) in enumerate(tqdm(data_loader_train)):
        samples = samples.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(args.norm_scale)

            # class embed
            class_embedding = model.class_emb(labels)

            # patchify and mask (drop) tokens
            x = model.patchify(x)
            gt_latents = x.clone().detach()
            orders = model.sample_orders(bsz=x.size(0))
            mask = model.random_masking(x, orders)

            # mae encoder
            x = model.forward_mae_encoder(x, mask, class_embedding)

            # mae decoder
            z = model.forward_mae_decoder(x, mask)
        
        optimizer.zero_grad()
        out = probe(z)
        loss = (out - gt_latents).pow(2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    torch.save(probe.state_dict(), os.path.join(args.output_dir, "probe.pth"))

else:
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    checkpoint = torch.load(os.path.join(args.output_dir, "probe.pth"), map_location='cpu')
    probe.load_state_dict(checkpoint, strict=True)
    probe.eval()
    print("Probe loaded from %s" % args.output_dir)

    with torch.no_grad():
        batch_size = 64
        classes = torch.randint(low=0, high=1000, size=(batch_size,)).cuda()
        # init and sample generation orders
        mask = torch.ones(batch_size, model.seq_len).cuda()
        tokens = torch.zeros(batch_size, model.seq_len, model.token_embed_dim).cuda()
        orders = model.sample_orders(bsz=batch_size)

        indices = list(range(args.num_iter))
        indices = tqdm(indices)

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            class_embedding = model.class_emb(classes).cuda()

            # mae encoder
            x = model.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = model.forward_mae_decoder(x, mask)

            # save_images(tokens, os.path.join(args.output_dir, "step_{}".format(step)))
            # # Use probe to predict the latents
            # out = probe(z)
            # completed_latents = out * mask.int().unsqueeze(-1) + tokens * (1 - mask.int().unsqueeze(-1))
            # save_images(completed_latents, os.path.join(args.output_dir, "probe_completed_{}".format(step)))
            # real_sampled_latents = model.next_layer_sample(z.reshape(-1, 768), temperature=1.0, cfg=1.0)
            # real_sampled_latents = real_sampled_latents.reshape(batch_size, model.seq_len, -1)
            # loss = (real_sampled_latents - completed_latents).pow(2).mean(-1)
            # loss = (loss * mask).sum() / mask.sum()
            # print(f"Step {step}, Loss: {loss.item()}")

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / args.num_iter)
            mask_len = torch.tensor([np.floor(model.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                        torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, batch_size, model.seq_len)
            if step >= args.num_iter - 1:
                mask_to_pred = mask[:batch_size].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:batch_size].bool(), mask_next.bool())
            mask = mask_next

            if not args.cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if args.cfg_schedule == "linear":
                cfg_iter = 1 + (args.cfg - 1) * (model.seq_len - mask_len[0]) / model.seq_len
            elif args.cfg_schedule == "constant":
                cfg_iter = args.cfg
            else:
                raise NotImplementedError

            sampled_token_latent = model.next_layer_sample(z, temperature=args.temperature, cfg=cfg_iter)

            if not args.cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

            sampled_token_latent = model.next_layer_sample(z[:128].repeat(5000, 1), temperature=args.temperature, cfg=cfg_iter)
            torch.save(sampled_token_latent, os.path.join(args.output_dir, "diff_sampled_latents/sampled_latents_{}_5000times.pth".format(step)))

        save_images(tokens, os.path.join(args.output_dir, "fully_sampled"))
