# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # text_encoder = TextEmbedder().to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)

    # Labels to condition the model with (feel free to change):

    # Create sampling noise:
    if args.use_fp16:
        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
    else:
        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    if using_cfg:
        z = torch.cat([z, z], 0)
        y = torch.randint(0, args.num_classes, (1,), device=device)
        y_null = torch.tensor([101] * 1, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
        sample_fn = model.forward_with_cfg
    else:
        sample_fn = model.forward
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)

    # Sample images:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

    print(samples.shape)
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)
    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b f c h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
    # Save and display images:

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)


    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
