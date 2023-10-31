import os
import json
import torch
import decord
import torchvision

import numpy as np

import random
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from torchvision import transforms
import traceback

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def numpy2tensor(x):
    return torch.from_numpy(x)


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


class FaceForensicsImages(torch.utils.data.Dataset):
    """Load the FaceForensics video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = get_filelist(configs.data_path)
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        self.video_length = len(self.video_lists)

        # ffs video frames
        self.video_frame_path = configs.frame_data_path
        self.video_frame_txt = configs.frame_data_txt
        self.video_frame_files = [frame_file.strip() for frame_file in open(self.video_frame_txt)]
        random.shuffle(self.video_frame_files)
        self.use_image_num = configs.use_image_num
        self.image_tranform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __getitem__(self, index):
        video_index = index % self.video_length
        path = self.video_lists[video_index]
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        video = vframes[frame_indice]
        # videotransformer data proprecess
        video = self.transform(video) # T C H W

        # get video frames
        images = []
        for i in range(self.use_image_num):
            while True:
                try:      
                    image = Image.open(os.path.join(self.video_frame_path, self.video_frame_files[index+i])).convert("RGB")
                    image = self.image_tranform(image).unsqueeze(0)
                    images.append(image)
                    break
                except Exception as e:
                    traceback.print_exc()
                    index = random.randint(0, len(self.video_frame_files) - self.use_image_num)
        images =  torch.cat(images, dim=0)
        
        assert len(images) == self.use_image_num

        video_cat = torch.cat([video, images], dim=0)

        return {'video': video_cat, 'video_name': 1}

    def __len__(self):
        return len(self.video_frame_files)


if __name__ == '__main__':
    import argparse
    import torchvision
    import video_transforms
    
    import torch.utils.data as Data
    import torchvision.transforms as transform
    
    from PIL import Image
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--use-image-num", type=int, default=5)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--dataset", type=str, default='webvideo10m')
    parser.add_argument("--test-run", type=bool, default='')
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/preprocessed_ffs/train/videos/")
    parser.add_argument("--frame-data-path", type=str, default="/path/to/datasets/preprocessed_ffs/train/images/")
    parser.add_argument("--frame-data-txt", type=str, default="/path/to/datasets/faceForensics_v1/train_list.txt")
    config = parser.parse_args()

    temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_webvideo = transform.Compose([
            video_transforms.ToTensorVideo(),
            transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    dataset = FaceForensicsImages(config, transform=transform_webvideo, temporal_sample=temporal_sample)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=4)

    for i, video_data in enumerate(dataloader):
        video, video_label = video_data['video'], video_data['video_name']
        # print(video_label)
        # print(image_label)
        print(video.shape)
        print(video_label)
        # video_ = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # print(video_.shape)
        # try:
        #     torchvision.io.write_video(f'./test/{i:03d}_{video_label}.mp4', video_[:16], fps=8)
        # except:
        #     pass
        
        # if i % 100 == 0 and i != 0:
        #     break
    print('Done!')