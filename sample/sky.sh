#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python sample/sample.py \
--config ./configs/sky/sky_sample.yaml \
--ckpt ./share_ckpts/skytimelapse.pt \
--save_video_path ./test