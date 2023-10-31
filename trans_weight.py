import torch
import argparse
from omegaconf import OmegaConf
from models import get_models


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample_ddp.yaml")
args = parser.parse_args()
args = OmegaConf.load(args.config)

args.latent_size = 32
model = get_models(args)

    # model_dict = unet.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {}
    # for k, v in state_dict.items():
    #     if k in model_dict:
    #         pretrained_dict[k] = v
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # unet.load_state_dict(model_dict)

model_path = '/mnt/lustre/maxin/work/Video-Generation-Transformers/lavita-github/share_ckpts/ucf101.pt'
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        

if "ema" in checkpoint:  # supports checkpoints from train.py
    print('Ema existing!')
    state_dict = checkpoint["ema"]

model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {}
for k, v in state_dict.items():
    if 'time_embed' in k:
        k = k.replace('time_embed', 'temp_embed')
    pretrained_dict[k] = v
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

checkpoint = {
    "ema": model.state_dict(),
}
checkpoint_path = "update_ucf101.pt"
torch.save(checkpoint, checkpoint_path)