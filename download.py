# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os

def find_model(model_name):
    """
    Finds a pre-trained LAVITA model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find LAVITA checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Ema existing!')
        checkpoint = checkpoint["ema"]
    return checkpoint