## Download datasets

Here are the links to download the datasets [FaceForensics](https://huggingface.co/datasets/maxin-cn/FaceForensics), [SkyTimelapse](https://huggingface.co/datasets/maxin-cn/SkyTimelapse/tree/main), [UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar), and [Taichi-HD](https://huggingface.co/datasets/maxin-cn/Taichi-HD).


## Dataset structure

All datasets follow their original dataset structure. As for video-image joint training, there is a `train_list.txt` file, whose format is `video_name/frame.jpg`. Here, we show an example of the FaceForensics datsset.

All datasets retain their original structure. For video-image joint training, there is a `train_list.txt` file formatted as `video_name/frame.jpg`. Below is an example from the FaceForensics dataset.

```bash
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000306.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000111.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000007.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000057.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000084.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000268.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000270.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000259.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000127.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000099.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000189.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000228.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000026.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000081.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000094.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000223.jpg
aS62n5PdTIU_1_8WGsQ0Y7uyU_1/000055.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000486.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000396.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000475.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000028.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000261.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000294.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000257.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000490.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000143.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000190.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000476.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000397.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000437.jpg
qEnKi82wWgE_2_rJPM8EdWShs_1/000071.jpg
```

## Evaluation

We follow [StyleGAN-V](https://github.com/universome/stylegan-v) to measure the quality of the generated video. The code for calculating the relevant metrics is located in [tools](../tools/) folder. To measure the quantitative metrics of your generated results, you need to put all the videos from real data into a folder, turn them into video frames and do `center-crop-resize-256` (the same goes for fake data). Then you can run the following command on one GPU:

```bash
# cd Latte
bash tools/eval_metrics.sh
```