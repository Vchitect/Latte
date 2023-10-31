export CUDA_VISIBLE_DEVICES=2
# python train.py --config configs/ucf101_train.yaml
# python train.py --config configs/ffs_train.yaml
# python train.py --config configs/ffs/ffs_wopre_model3.yaml
# python train.py --config configs/taichi/taichi_wopre_train.yaml
# python train.py --config configs/ffs/ffs_wopre_model4.yaml
# python train.py --config configs/ffs/ffs_wopre_model1r.yaml
# python train.py --config configs/ffs/ffs_wopre_model2.yaml
# python train.py --config configs/webvid/webv2m_wopre_train_v2.yaml
python train_with_img.py --config configs/webvid/web2m_wopre_img_train_v2.yaml
# python train.py --config configs/sky/sky_wopre_train.yaml
# python train.py --config configs/ucf101/ucf101_wopre_train.yaml
# python train_with_img.py --config configs/sky/sky_wopre_img_train.yaml
# python train_with_img.py --config configs/ucf101/ucf101_wopre_img_train_nl.yaml
# python train_with_img.py --config configs/taichi/taichi_wopre_img_train.yaml
# TORCH_DISTRIBUTED_DEBUG=DETAIL python train_with_img.py --config configs/ucf101/ucf101_wopre_img_train_may.yaml
# python train_with_img.py --config configs/ffs/ffs_wopre_img_model1.yaml