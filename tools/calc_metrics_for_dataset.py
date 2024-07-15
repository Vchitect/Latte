# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import sys; sys.path.extend(['.', 'tools'])
import os
import click
import tempfile
import torch
from omegaconf import OmegaConf
from tools import dnnlib

from metrics import metric_main
from metrics import metric_utils
from tools.torch_utils import training_stats
from tools.torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric,
            dataset_kwargs=args.dataset_kwargs, # real
            gen_dataset_kwargs=args.gen_dataset_kwargs, # fake
            generator_as_dataset=args.generator_as_dataset,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
            cache=args.use_cache,
            num_runs=args.num_runs,
        )

        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir)

        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

def calc_metrics_for_dataset(ctx, metrics, real_data_path, fake_data_path, mirror, resolution, gpus, verbose, use_cache: bool, num_runs: int):
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    dummy_dataset_cfg = OmegaConf.create({'max_num_frames': 10000})

    # Initialize dataset options for real data.
    args.dataset_kwargs = dnnlib.EasyDict(
        class_name='utils.dataset.VideoFramesFolderDataset',
        path=real_data_path,
        cfg=dummy_dataset_cfg,
        xflip=mirror,
        resolution=resolution,
        use_labels=False,
    )

    # Initialize dataset options for fake data.
    args.gen_dataset_kwargs = dnnlib.EasyDict(
        class_name='utils.dataset.VideoFramesFolderDataset',
        path=fake_data_path,
        cfg=dummy_dataset_cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
    )
    args.generator_as_dataset = True

    # Print dataset options.
    if args.verbose:
        print('Real data options:')
        print(args.dataset_kwargs)

        print('Fake data options:')
        print(args.gen_dataset_kwargs)

    print('*' * 50 + 'parting line' + '*' * 50)
    print('Fake data options:')
    print(args.gen_dataset_kwargs)

    # Locate run dir.
    args.run_dir = None
    args.use_cache = use_cache
    args.num_runs = num_runs

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fvd2048_16f,fid50k_full', show_default=True)
@click.option('--real_data_path', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
@click.option('--fake_data_path', help='Generated images (directory or zip)', metavar='PATH')
@click.option('--mirror', help='Should we mirror the real data?', type=bool, metavar='BOOL')
@click.option('--resolution', help='Resolution for the source dataset', type=int, metavar='INT')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--use_cache', help='Use stats cache', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--num_runs', help='Number of runs', type=int, default=1, metavar='INT', show_default=True)
def calc_metrics_cli_wrapper(ctx, *args, **kwargs):
    calc_metrics_for_dataset(ctx, *args, **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics_cli_wrapper() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
