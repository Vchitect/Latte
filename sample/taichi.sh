#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python sample/sample.py \
--config ./configs/ucf101/taichi_sample.yaml \
--ckpt  ./share_ckpts/taichi-hd.pt \
--save_video_path ./test