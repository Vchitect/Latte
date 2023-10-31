import os
import torch
import random
import torch.utils.data as data
import numpy as np
import copy
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class SkyImages(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all, self.video_frame_all = self.load_video_frames(self.data_path)

        # sky video frames
        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num

    def __getitem__(self, index):

        video_index = index % self.video_num
        vframes = self.data_all[video_index]
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int) # start, stop, num=50

        select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.frame_interval] 

        video_frames = []
        for path in select_video_frames:
            video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        # get video frames
        images = []

        for i in range(self.use_image_num):
            while True:
                try:      
                    video_frame_path = self.video_frame_all[index+i]
                    image = torch.as_tensor(np.array(Image.open(video_frame_path), dtype=np.uint8, copy=True)).unsqueeze(0)
                    images.append(image)
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)

        images =  torch.cat(images, dim=0).permute(0, 3, 1, 2)
        images = self.transform(images)
        assert len(images) == self.use_image_num

        video_cat = torch.cat([video_clip, images], dim=0)

        return {'video': video_cat, 'video_name': 1}

    def __len__(self):
        return self.video_frame_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0]) # root
                print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.target_video_len * self.frame_interval): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
                for frame in frames:
                    frames_all.append(frame)
        self.video_num = len(data_all)
        self.video_frame_num = len(frames_all)
        return data_all, frames_all
    

if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/sky_timelapse/sky_train/")
    parser.add_argument("--use-image-num", type=int, default=5)
    config = parser.parse_args()

    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        # video_transforms.CenterCropVideo(256),
        video_transforms.CenterCropResizeVideo(256),
        # video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    taichi_dataset = SkyImages(config, transform=trans, temporal_sample=temporal_sample)
    print(len(taichi_dataset))
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, video_data in enumerate(taichi_dataloader):
        print(video_data['video'].shape)
        
        # print(video_data.dtype)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        # video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # torchvision.io.write_video('./test_data' + 'test.mp4', video_, fps=8)
        # exit()