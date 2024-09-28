## Latte: Latent Diffusion Transformer for Video Generation<br><sub>Official PyTorch Implementation</sub>

<!-- ### [Paper](https://arxiv.org/abs/2401.03048v1) | [Project Page](https://maxin-cn.github.io/latte_project/) -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2401.03048-b31b1b.svg)](https://arxiv.org/abs/2401.03048) -->
[![arXiv](https://img.shields.io/badge/arXiv-2401.03048-b31b1b.svg)](https://arxiv.org/abs/2401.03048)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://maxin-cn.github.io/latte_project/)
[![HF Demo](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/UOeU0ywVxl7) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/maxin-cn/Latte-1)
[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/RguYqhVU92)

[![Static Badge](https://img.shields.io/badge/Latte--1%20checkpoint%20(T2V)-HuggingFace-yellow?logoColor=violet%20Latte-1%20checkpoint)](https://huggingface.co/maxin-cn/Latte-1)
[![Static Badge](https://img.shields.io/badge/Latte%20checkpoint%20-HuggingFace-yellow?logoColor=violet%20Latte%20checkpoint)](https://huggingface.co/spaces/maxin-cn/Latte-1)

This repo contains PyTorch model definitions, pre-trained weights, training/sampling code and evaluation code for our paper 
Latte: Latent Diffusion Transformer for Video Generation. 

> [**Latte: Latent Diffusion Transformer for Video Generation**](https://maxin-cn.github.io/latte_project/)<br>
> [Xin Ma](https://maxin-cn.github.io/), [Yaohui Wang*](https://wyhsirius.github.io/), [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ), [Gengyun Jia](https://scholar.google.com/citations?user=_04pkGgAAAAJ&hl=zh-CN), [Ziwei Liu](https://liuziwei7.github.io/), [Yuan-Fang Li](https://users.monash.edu/~yli/), [Cunjian Chen](https://cunjian.github.io/), [Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ&hl=zh-CN)
> (*Corresponding Author & Project Lead)
<!-- > <br>Monash University, Shanghai Artificial Intelligence Laboratory,<br> NJUPT, S-Lab, Nanyang Technological University 

We propose a novel Latent Diffusion Transformer, namely Latte, for video generation. Latte first extracts spatio-temporal tokens from input videos and then adopts a series of Transformer blocks to model video distribution in the latent space. In order to model a substantial number of tokens extracted from videos, four efficient variants are introduced from the perspective of decomposing the spatial and temporal dimensions of input videos. To improve the quality of generated videos, we determine the best practices of Latte through rigorous experimental analysis, including video clip patch embedding, model variants, timestep-class information injection, temporal positional embedding, and learning strategies. Our comprehensive evaluation demonstrates that Latte achieves state-of-the-art performance across four standard video generation datasets, i.e., FaceForensics, SkyTimelapse, UCF101, and Taichi-HD. In addition, we extend Latte to text-to-video generation (T2V) task, where Latte achieves comparable results compared to recent T2V models. We strongly believe that Latte provides valuable insights for future research on incorporating Transformers into diffusion models for video generation.

 ![The architecture of Latte](visuals/architecture.svg){width=20}
 -->

<!--
<div align="center">
    <img src="visuals/architecture.svg" width="650">
</div>

This repository contains:

* 🪐 A simple PyTorch [implementation](models/latte.py) of Latte
* ⚡️ **Pre-trained Latte models** trained on FaceForensics, SkyTimelapse, Taichi-HD and UCF101 (256x256). In addition, we provide a T2V checkpoint (512x512). All checkpoints can be found [here](https://huggingface.co/maxin-cn/Latte/tree/main). 

* 🛸 A Latte [training script](train.py) using PyTorch DDP.
-->

<video controls loop src="https://github.com/Vchitect/Latte/assets/7929326/a650cd84-2378-4303-822b-56a441e1733b" type="video/mp4"></video>

## News
- (🔥 New) **Jul 11, 2024** 💥 **Latte-1 is now integrated into [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/latte). Thanks to [@yiyixuxu](https://github.com/yiyixuxu), [@sayakpaul](https://github.com/sayakpaul), [@a-r-r-o-w](https://github.com/a-r-r-o-w) and [@DN6](https://github.com/DN6).** You can easily run Latte using the following code. We also support inference with 4/8-bit quantization, which can reduce GPU memory from 17 GB to 9 GB. Please refer to this [tutorial](docs/latte_diffusers.md) for more information.

```
# Please update the version of diffusers at leaset to 0.30.0
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from torchvision.utils import save_image
import torch
import imageio

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
video_length = 16 # 1 (text-to-image) or 16 (text-to-video)
pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16).to(device)

# Using temporal decoder of VAE
vae = AutoencoderKLTemporalDecoder.from_pretrained("maxin-cn/Latte-1", subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
pipe.vae = vae

prompt = "a cat wearing sunglasses and working as a lifeguard at pool."
videos = pipe(prompt, video_length=video_length, output_type='pt').frames.cpu()
```

- (🔥 New) **Jun 26, 2024** 💥 Latte is supported by [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), which is a user-friendly, high-performance infrastructure for video generation.

- (🔥 New) **May 23, 2024** 💥 **Latte-1** is released! Pre-trained model can be downloaded [here](https://huggingface.co/maxin-cn/Latte-1/tree/main/transformer). **We support both T2V and T2I**. Please run `bash sample/t2v.sh` and `bash sample/t2i.sh` respectively.

<!--
<div align="center">
    <img src="visuals/latteT2V.gif" width=88%>
</div>
-->

- (🔥 New) **Feb 24, 2024** 💥 We are very grateful that researchers and developers like our work. We will continue to update our LatteT2V model, hoping that our efforts can help the community develop. Our Latte discord channel <a href="https://discord.gg/RguYqhVU92" style="text-decoration:none;">
<img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a> is created for discussions. Coders are welcome to contribute.

- (🔥 New) **Jan 9, 2024** 💥 An updated LatteT2V model initialized with the [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) is released, the checkpoint can be found [here](https://huggingface.co/maxin-cn/Latte-0/tree/main/transformer).

- (🔥 New) **Oct 31, 2023** 💥 The training and inference code is released. All checkpoints (including FaceForensics, SkyTimelapse, UCF101, and Taichi-HD) can be found [here](https://huggingface.co/maxin-cn/Latte/tree/main). In addition, the LatteT2V inference code is provided.


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/Vchitect/Latte
cd Latte
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate latte
```


## Sampling 

You can sample from our **pre-trained Latte models** with [`sample.py`](sample/sample.py). Weights for our pre-trained Latte model can be found [here](https://huggingface.co/maxin-cn/Latte).  The script has various arguments to adjust sampling steps, change the classifier-free guidance scale, etc. For example, to sample from our model on FaceForensics, you can use:

```bash
bash sample/ffs.sh
```

or if you want to sample hundreds of videos, you can use the following script with Pytorch DDP:

```bash
bash sample/ffs_ddp.sh
```

If you want to try generating videos from text, just run `bash sample/t2v.sh`. All related checkpoints will download automatically.

If you would like to measure the quantitative metrics of your generated results, please refer to [here](docs/datasets_evaluation.md).

## Training

We provide a training script for Latte in [`train.py`](train.py). The structure of the datasets can be found [here](docs/datasets_evaluation.md). This script can be used to train class-conditional and unconditional
Latte models. To launch Latte (256x256) training with `N` GPUs on the FaceForensics dataset 
:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --config ./configs/ffs/ffs_train.yaml
```

or If you have a cluster that uses slurm, you can also train Latte's model using the following scripts:

 ```bash
sbatch slurm_scripts/ffs.slurm
```

We also provide the video-image joint training scripts [`train_with_img.py`](train_with_img.py). Similar to [`train.py`](train.py) scripts, these scripts can be also used to train class-conditional and unconditional
Latte models. For example, if you want to train the Latte model on the FaceForensics dataset, you can use:

```bash
torchrun --nnodes=1 --nproc_per_node=N train_with_img.py --config ./configs/ffs/ffs_img_train.yaml
```

If you are familiar with `PyTorch Lightning`, you can also use the training script [`train_pl.py`](train_pl.py) and [`train_with_img_pl.py`](train_with_img_pl.py) provided by [@zhang.haojie](https://github.com/zhang-haojie),

```bash
python train_pl.py --config ./configs/ffs/ffs_train.yaml
```

or

```bash
python train_with_img_pl.py --config ./configs/ffs/ffs_img_train.yaml
```

This script automatically detects available GPUs and uses distributed training.

## Contact Us
**Yaohui Wang**: [wangyaohui@pjlab.org.cn](mailto:wangyaohui@pjlab.org.cn)
**Xin Ma**: [xin.ma1@monash.edu](mailto:xin.ma1@monash.edu)

## Citation
If you find this work useful for your research, please consider citing it.
```bibtex
@article{ma2024latte,
  title={Latte: Latent Diffusion Transformer for Video Generation},
  author={Ma, Xin and Wang, Yaohui and Jia, Gengyun and Chen, Xinyuan and Liu, Ziwei and Li, Yuan-Fang and Chen, Cunjian and Qiao, Yu},
  journal={arXiv preprint arXiv:2401.03048},
  year={2024}
}
```


## Acknowledgments
Latte has been greatly inspired by the following amazing works and teams: [DiT](https://github.com/facebookresearch/DiT) and [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha), we thank all the contributors for open-sourcing.


## License
The code and model weights are licensed under [LICENSE](LICENSE).
