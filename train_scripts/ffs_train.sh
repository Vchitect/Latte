export CUDA_VISIBLE_DEVICES=5
# torchrun --nnodes=1 --nproc_per_node=2 --master_port=29509 train.py --config ./configs/ffs/ffs_train.yaml
python train.py --config ./configs/ffs/ffs_train.yaml