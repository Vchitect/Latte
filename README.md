## LAVITA: Latent Video Diffusion Models with Spatio-temporal Transformers (LAVITA)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://maxin-cn.github.io/lavita_project/) | [Project Page](https://maxin-cn.github.io/lavita_project/)



This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring 
latent diffusion models with transformers (LAVITA). You can find more visualizations on our [project page](https://maxin-cn.github.io/lavita_project/).

> [**LAVITA: Latent Video Diffusion Models with Spatio-temporal Transformers**](https://maxin-cn.github.io/lavita_project/)<br>
> [Xin Ma](https://maxin-cn.github.io/), [Yaohui Wang](https://wyhsirius.github.io/), [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ), [Yuan-Fang Li](https://users.monash.edu/~yli/), [Cunjian Chen](https://cunjian.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ&hl=zh-CN)
> <br>Department of Data Science \& AI, Faculty of Information Technology, Monash University <br> Shanghai Artificial Intelligence Laboratory, S-Lab, Nanyang Technological University<br>

 We propose a novel architecture, the latent video diffusion model with spatio-temporal transformers, referred to as LAVITA, which integrates the Transformer architecture into diffusion models for the first time within the realm of video generation. Conceptually, LATIVA models spatial and temporal information separately to accommodate their inherent disparities as well as to reduce the computational complexity. Following this design strategy, we design several Transformer-based model variants to integrate spatial and temporal information harmoniously. Moreover, we identify the best practices in architectural choices and learning strategies for LAVITA through rigorous empirical analysis. Our comprehensive evaluation demonstrates that LAVITA achieves state-of-the-art performance across several standard video generation benchmarks, including FaceForensics, SkyTimelapse, UCF101, and Taichi-HD, outperforming current best models.

 ![The architecure of LAVITA](visuals/architecture.svg)

This repository contains:

* 🪐 A simple PyTorch [implementation](models/lavita.py.py) of LAVITA
* ⚡️ Pre-trained LAVITA models trained on FaceForensics, SkyTimelapse, Taichi-HD and UCF101 (256x256)

* 🛸 A LAVITA [training script](train.py) using PyTorch DDP



## Setup

First, download and set up the repo:

```bash
git clone https://github.com/maxin-cn/LAVITA.git
cd LAVITA
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate lavita
```


## Sampling 

**Pre-trained LAVITA checkpoints.** You can sample from our pre-trained LAVITA models with [`sample.py`](sample/sample.py). Weights for our pre-trained LAVITA model can be found [here](https://huggingface.co/maxin-cn/LAVITA). The script has various arguments to adjust sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our model on FaceForensics, you can use:

```bash
bash sample/ffs.sh
```

or if you want to sample hundreds of videos, you can use the following script with Pytorch DDP:

```bash
bash sample/ffs_ddp.sh
```

## Training LAVITA

We provide a training script for LAVITA in [`train.py`](train.py). This script can be used to train class-conditional and unconditional
LAVITA models. To launch LAVITA (256x256) training with `N` GPUs on the FaceForensics dataset 
:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --config ./configs/ffs/ffs_train.yaml
```

or If you have a cluster that uses slurm, you can also train LAVITA's model using the following scripts:

 ```bash
sbatch slurm_scripts/ffs.slurm
```

We also provide the video-image joint training scripts [`train_with_img.py`](train_with_img.py). Similar to [`train.py`](train.py) scripts, this scripts can be also used to train class-conditional and unconditional
LAVITA models. For example, if you wan to train LAVITA model on the FaceForensics dataset, you can use:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --config ./configs/ffs/ffs_img_train.yaml
```

<!-- ## BibTeX

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
``` -->


## Acknowledgments
Video generation models are improving quickly and the development of LAVITA has been greatly inspired by the following amazing works and teams: [DiT](https://github.com/facebookresearch/DiT), [U-ViT](https://github.com/baofff/U-ViT), and [Tune-A-Video](https://github.com/showlab/Tune-A-Video).


## License
The code and model weights are licensed under [CC-BY-NC](license_for_usage.txt).
