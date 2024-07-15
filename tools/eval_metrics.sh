export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
--real_data_path /path/to/real_data//images \
--fake_data_path /path/to/fake_data/images \
--mirror 1 --gpus 1 --resolution 256 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0