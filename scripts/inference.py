import os
import io
import platform
import re
import gc
import argparse
import warnings
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import av
import cv2
import uuid
import imageio
import base64
import shutil

import numpy as np
import torch
import torchvision
from PIL import Image
import decord
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import LattePipeline


def flush():
    gc.collect()
    torch.cuda.empty_cache()

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024
    
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File deleted successfully")
    else:
        print("File not found.")

def base64_to_video(base64_string=""):
    """Converts a Base64 image string to a PyTorch tensor"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img) 

    # Convert to tensor - PyTorch generally expects channels first (e.g., (C, H, W))
    file_output_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return file_output_tensor

def video_to_base64(src_path="", delete_src=False):
    try:
        with open(src_path, "rb") as video_file:
            video_data = video_file.read()
            base64_encoded_data = base64.b64encode(video_data)
            if delete_src:
                delete_file(src_path)
            return base64_encoded_data.decode('utf-8')
    except:
        return None

def video_to_tensor(video_path="", output_format="TCHW"):
    video_tensor = None
    video_tensor, _, info = torchvision.io.read_video(video_path, pts_unit = "sec", output_format=output_format)
    return video_tensor

def tensor_to_video(video_tensor, output_filename="mp_video.mp4", fps=1):
    if video_tensor.shape[-1] > 3:
        video_tensor = video_tensor.permute(0, 2, 3, 1)
    
    height, width = video_tensor.shape[1], video_tensor.shape[2]
    container = av.open(output_filename, mode='w', format='mp4')
    stream = container.add_stream('libx264', rate=fps)  # Common codec (H.264)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'  # Set a suitable pixel format
    for i in range(video_tensor.shape[0]):
        # frame = video_tensor[i].permute(1, 2, 0).numpy()  # PyTorch => OpenCV compatible
        frame = video_tensor[i].numpy().astype('uint8')  # PyTorch => OpenCV compatible
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV usually expects BGR

        av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        for packet in stream.encode(av_frame):
            container.mux(packet)
            
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return output_filename

# def to_video(fn: str, frames: list[np.ndarray], fps: int):
def to_video(fn: str, frames: any, fps: int) -> str:
    writer = imageio.get_writer(fn, format='FFMPEG', fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return fn

def initialize_pipeline(
    model_id: str,
    device: str = "cuda",
    load_in_4bit: int = True,
):
    text_encoder = None
    if load_in_4bit:
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
    else:
        pipe = LattePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

        # Using temporal decoder of VAE
        vae = AutoencoderKLTemporalDecoder.from_pretrained(model_id, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
        pipe.vae = vae

    return pipe, text_encoder



def inference_function( args ):
    decord.bridge.set_bridge("torch")
    flush()

    args.model = args.model if hasattr(args, 'model') else "maxin-cn/Latte-1"
    args.prompt = args.prompt if hasattr(args, 'prompt') else None
    args.negative_prompt = args.negative_prompt if hasattr(args, 'negative_prompt') else None
    args.num_steps = args.num_steps if hasattr(args, 'num_steps') else 16
    args.num_frames = args.num_frames if hasattr(args, 'num_frames') else 16
    args.fps = args.fps if hasattr(args, 'fps') else 4
    args.quantize = args.quantize if hasattr(args, 'quantize') else True
    args.device = args.device if hasattr(args, 'device') else "cuda"
    args.seed = args.seed if hasattr(args, 'seed') else 0
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else 0


    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    torch.manual_seed(args.seed)
    pipe, text_encoder = initialize_pipeline(model_id=args.model, device=args.device, load_in_4bit=args.quantize)
    if args.quantize is not True:
        videos = pipe(args.prompt, video_length=args.num_frames, output_type='pt').frames.cpu()
    else:
        with torch.no_grad():
            neg_prompt = args.negative_prompt if args.negative_prompt is not None else ""
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(args.prompt, negative_prompt=neg_prompt )

        del text_encoder
        del pipe
        flush()

        pipe = LattePipeline.from_pretrained(
            args.model,
            text_encoder=None,
            torch_dtype=torch.float16,
        ).to(args.device)

        videos = pipe(
            video_length=args.num_frames,
            negative_prompt=args.negative_prompt, 
            num_inference_steps=args.num_steps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type="pt",
        ).frames.cpu()
    print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")


    output_path = f"{args.output_dir}/latte_output_${(uuid.uuid4().hex)[:4]}"
    if args.num_frames > 1:
        output_path = f"{output_path}.mp4"
        videos_uint8 = (videos.clamp(0, 1) * 255).to(dtype=torch.uint8) # convert to uint8
        imageio.mimwrite(output_path, videos_uint8[0].permute(0, 2, 3, 1), fps=8, quality=5) # highest quality is 10, lowest is 0
    else:
        output_path = f"{output_path}.png"
        save_image(videos[0], output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="The Model name or Model id from repository.")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on.")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against.")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate.")
    parser.add_argument("-f", "--fps", type=int, default=4, help="FPS of output video.")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-q", "--quantize", type=bool, default=True, help="Whether to run the quantized version of the Model.")
    parser.add_argument("-r", "--seed", type=int, default=0, help="Random seed to make generations reproducible.")
    parser.add_argument("-o", "--output-dir", type=str, default="./outputs", help="Directory to save output video to.")
    args = parser.parse_args()

    inference_function(args)