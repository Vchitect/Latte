#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29520 sample/sample_ddp.py \
--config ./configs/ucf101/ucf101_sample.yaml \
--ckpt  ./share_ckpts/ucf101.pt \
--save_video_path ./test
