import os
import argparse

import torch

from util.loader import CachedFolder
from models import mar


def get_args_parser():
    parser = argparse.ArgumentParser('Train a probe on MAR', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

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

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser

parser = get_args_parser()
args = parser.parse_args()

dataset_train = CachedFolder(args.cached_path)

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
# Convert diffloss to arhead
checkpoint['model'] = {
    k.replace('diff_loss', 'ar_gmm'): v for k, v in checkpoint['model'].items()
}
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
print("Model loaded from %s" % args.resume)

##TODO