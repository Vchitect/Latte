"""
Frechet Video Distance (FVD). Matches the original tensorflow implementation from
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
up to the upsampling operation. Note that this tf.hub I3D model is different from the one released in the I3D repo.
"""

import copy
import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

NUM_FRAMES_IN_BATCH = {128: 128, 256: 128, 512: 64, 1024: 32}

#----------------------------------------------------------------------------

def compute_fvd(opts, max_real: int, num_gen: int, num_frames: int, realdata_subsample_factor: int=3, gendata_subsample_factor: int=1):
    # Perfectly reproduced torchscript version of the I3D model, trained on Kinetics-400, used here:
    # https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
    # Note that the weights on tf.hub (used in the script above) differ from the original released weights
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    # real data args
    opts = copy.deepcopy(opts)
    opts.dataset_kwargs.load_n_consecutive = num_frames
    # opts.dataset_kwargs.load_n_consecutive = None
    opts.dataset_kwargs.subsample_factor = realdata_subsample_factor
    opts.dataset_kwargs.discard_short_videos = True
    batch_size = NUM_FRAMES_IN_BATCH[opts.dataset_kwargs.resolution] // num_frames

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=0,
        capture_mean_cov=True, max_items=max_real, temporal_detector=True, batch_size=batch_size).get_mean_cov()

    if opts.generator_as_dataset:
        # fake data args
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_opts.dataset_kwargs.load_n_consecutive = num_frames
        gen_opts.dataset_kwargs.load_n_consecutive_random_offset = False
        gen_opts.dataset_kwargs.subsample_factor = gendata_subsample_factor
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=gendata_subsample_factor)

    mu_gen, sigma_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=1, capture_mean_cov=True,
        max_items=num_gen, temporal_detector=True, batch_size=batch_size, **gen_kwargs).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
