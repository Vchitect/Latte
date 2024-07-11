## Requirements

Please follow [README](../README.md) to install the environment. After installation, update the version of diffusers at leaset to 0.30.0.

## Inference

```bash
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder

from torchvision.utils import save_image

import torch
import imageio

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
video_length = 1
pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16).to(device)

# if you want to use the temporal decoder of VAE, please uncomment the following codes
# vae = AutoencoderKLTemporalDecoder.from_pretrained("maxin-cn/Latte-1", subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
# pipe.vae = vae

prompt = "a cat wearing sunglasses and working as a lifeguard at pool."
videos = pipe(prompt, video_length=video_length, output_type='pt').frames.cpu()

if video_length > 1:
    videos = (videos.clamp(0, 1) * 255).to(dtype=torch.uint8) # convert to uint8
    imageio.mimwrite('./latte_output.mp4', videos[0].permute(0, 2, 3, 1), fps=8, quality=5) # highest quality is 10, lowest is 0
else:
    save_image(videos[0], './latte_output.png')
```

