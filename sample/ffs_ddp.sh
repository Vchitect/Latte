#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

torchrun --nnodes=1 --nproc_per_node=2 sample/sample_ddp.py \
--config ./configs/ffs/ffs_sample.yaml \
--ckpt ./share_ckpts/ffs.pt \
--save_video_path ./test
