## Requirements

Please follow [README](../README.md) to install the environment. After installation, update the version of `diffusers` at leaset to 0.30.0.

## Inference

```bash
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder

from torchvision.utils import save_image

import torch
import imageio

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
video_length = 1 # 1 or 16
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

## Inference with 4/8-bit quantization
[@Aryan](https://github.com/a-r-r-o-w) provides a quantization solution for inference, which can reduce GPU memory from 17 GB to 9 GB. Note that please install `bitsandbytes` (`pip install bitsandbytes`).

```bash
import gc

import torch
from diffusers import LattePipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import imageio
from torchvision.utils import save_image

torch.manual_seed(0)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

video_length = 16
model_id = "maxin-cn/Latte-1"

text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
)
pipe = LattePipeline.from_pretrained(
    model_id, 
    text_encoder=text_encoder,
    transformer=None,
    device_map="balanced",
)

with torch.no_grad():
    prompt = "a cat wearing sunglasses and working as a lifeguard at pool."
    negative_prompt = ""
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

del text_encoder
del pipe
flush()

pipe = LattePipeline.from_pretrained(
    model_id,
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")
# pipe.enable_vae_tiling()
# pipe.enable_vae_slicing()

videos = pipe(
    video_length=video_length,
    num_inference_steps=50,
    negative_prompt=None, 
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    output_type="pt",
).frames.cpu()

print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")

if video_length > 1:
    videos = (videos.clamp(0, 1) * 255).to(dtype=torch.uint8) # convert to uint8
    imageio.mimwrite('./latte_output.mp4', videos[0].permute(0, 2, 3, 1), fps=8, quality=5) # highest quality is 10, lowest is 0
else:
    save_image(videos[0], './latte_output.png')
```