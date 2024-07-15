# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

NUM_FRAMES_IN_BATCH = {128: 32, 256: 32, 512: 8, 1024: 2}

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    batch_size = NUM_FRAMES_IN_BATCH[opts.dataset_kwargs.resolution]

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, use_image_dataset=True).get_mean_cov()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_kwargs = dict(use_image_dataset=True)
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict()

    mu_gen, sigma_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs, batch_size=batch_size,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, **gen_kwargs).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
