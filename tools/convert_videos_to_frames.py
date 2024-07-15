"""
Converts a dataset of mp4 videos into a dataset of video frames
I.e. a directory of mp4 files becomes a directory of directories of frames
This speeds up loading during training because we do not need
"""
import os
from typing import List
import argparse
from pathlib import Path
from multiprocessing import Pool
from collections import Counter

from PIL import Image
import torchvision.transforms.functional as TVF
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def convert_videos_to_frames(source_dir: os.PathLike, target_dir: os.PathLike, num_workers: int, video_ext: str, **process_video_kwargs):
    broken_clips_dir = f'{target_dir}_broken_clips'
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(broken_clips_dir, exist_ok=True)

    clips_paths = [cp for cp in listdir_full_paths(source_dir) if cp.endswith(video_ext)]
    clips_fps = []
    tasks_kwargs = [dict(
        clip_path=cp,
        target_dir=target_dir,
        broken_clips_dir=broken_clips_dir,
        **process_video_kwargs,
     ) for cp in clips_paths]
    pool = Pool(processes=num_workers)

    for fps in tqdm(pool.imap_unordered(task_proxy, tasks_kwargs), total=len(clips_paths)):
        clips_fps.append(fps)

    print(f'All possible fps: {Counter(clips_fps).most_common()}')


def task_proxy(kwargs):
    """I do not know, how to pass several arguments to a pool job..."""
    return process_video(**kwargs)


def process_video(
    clip_path: os.PathLike, target_dir: os.PathLike, force_fps: int=None, target_size: int=None,
    broken_clips_dir: os.PathLike=None, compute_fps_only: bool=False) -> int:

    clip_name = os.path.basename(clip_path)
    clip_name = clip_name[:clip_name.rfind('.')]

    try:
        clip = VideoFileClip(clip_path)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f'Coudnt process clip: {clip_path}')
        if not broken_clips_dir is None:
            Path(os.path.join(broken_clips_dir, clip_name)).touch()
        return 0

    if compute_fps_only:
        return clip.fps

    fps = clip.fps if force_fps is None else force_fps
    clip_target_dir = os.path.join(target_dir, clip_name)
    clip_target_dir = clip_target_dir.replace('#', '_')
    os.makedirs(clip_target_dir, exist_ok=True)

    frame_idx = 0
    for frame in clip.iter_frames(fps=fps):
        frame = Image.fromarray(frame)
        h, w = frame.size[0], frame.size[1]
        min_size = min(h, w)
        if not target_size is None:
            # frame = TVF.resize(frame, size=target_size, interpolation=Image.LANCZOS)
            # frame = TVF.center_crop(frame, output_size=(target_size, target_size))
            frame = TVF.center_crop(frame, output_size=(min_size, min_size))
            frame = TVF.resize(frame, size=target_size, interpolation=Image.LANCZOS)
        frame.save(os.path.join(clip_target_dir, f'{frame_idx:06d}.jpg'), q=95)
        frame_idx += 1

    return clip.fps


def listdir_full_paths(d) -> List[os.PathLike]:
    return sorted([os.path.join(d, x) for x in os.listdir(d)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a dataset of mp4 files into a dataset of individual frames')
    parser.add_argument('-s', '--source_dir', type=str, help='Path to the source dataset')
    parser.add_argument('-t', '--target_dir', type=str, help='Where to save the new dataset')
    parser.add_argument('--video_ext', type=str, default='avi', help='Video extension')
    parser.add_argument('--target_size', type=int, default=128, help='What size should we resize to?')
    parser.add_argument('--force_fps', type=int, help='What fps should we run videos with?')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--compute_fps_only', action='store_true', help='Should we just compute fps?')
    args = parser.parse_args()

    convert_videos_to_frames(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_size=args.target_size,
        force_fps=args.force_fps,
        num_workers=args.num_workers,
        video_ext=args.video_ext,
        compute_fps_only=args.compute_fps_only,
    )
