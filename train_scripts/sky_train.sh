export CUDA_VISIBLE_DEVICES=4,5
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29509 train.py --config ./configs/sky/sky_train.yaml