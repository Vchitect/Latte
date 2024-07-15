# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import torch
import numpy as np
from tools import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import inception_score
from . import video_inception_score
from . import frechet_video_distance

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

def is_power_of_two(n: int) -> bool:
    return (n & (n-1) == 0) and n != 0

#----------------------------------------------------------------------------

def calc_metric(metric, num_runs: int=1, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    all_runs_results = [_metric_dict[metric](opts) for _ in range(num_runs)]
    total_time = time.time() - start_time

    # Broadcast results.
    for results in all_runs_results:
        for key, value in list(results.items()):
            if opts.num_gpus > 1:
                value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
                torch.distributed.broadcast(tensor=value, src=0)
                value = float(value.cpu())
            results[key] = value

    if num_runs > 1:
        results = {f'{key}_run{i+1:02d}': value for i, results in enumerate(all_runs_results) for key, value in results.items()}
        for key, value in all_runs_results[0].items():
            all_runs_values = [r[key] for r in all_runs_results]
            results[f'{key}_mean'] = np.mean(all_runs_values)
            results[f'{key}_std'] = np.std(all_runs_values)
    else:
        results = all_runs_results[0]

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)


@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

@register_metric
def fvd2048_16f(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16)
    return dict(fvd2048_16f=fvd)

@register_metric
def fvd2048_128f(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=128)
    return dict(fvd2048_128f=fvd)

@register_metric
def fvd2048_128f_subsample8f(opts):
    """Similar to `fvd2048_128f`, but we sample each 8-th frame"""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fvd = frechet_video_distance.compute_fvd(opts, max_real=2048, num_gen=2048, num_frames=16, subsample_factor=8)
    return dict(fvd2048_128f_subsample8f=fvd)

@register_metric
def isv2048_ucf(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = video_inception_score.compute_isv(opts, num_gen=2048, num_splits=10, backbone='c3d_ucf101')
    return dict(isv2048_ucf_mean=mean, isv2048_ucf_std=std)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

#----------------------------------------------------------------------------
