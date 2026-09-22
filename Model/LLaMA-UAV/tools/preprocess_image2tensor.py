"""
把每条轨迹的多视角原始RGB图像，批量预处理成CLIP要求的标准个是，打包成一个tensor文件
"""
import argparse
import multiprocessing
import torch
import os
import tqdm
import numpy as np
from transformers.models.clip import CLIPImageProcessor
from typing import Dict, Optional, Sequence, List

from PIL import Image

RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']

def arg_parse():
  parser = argparse.ArgumentParser(description="split video clip")
  parser.add_argument("--root_dir",
                      default='/path/to/your/dataset',
                      help='path to your dataset root dir')
  parser.add_argument("--map_list",
                        default=['NewYorkCity', 'ModernCityMap', 'NYCEnvironmentMegapa', 'TropicalIsland', 'ModularPark', 'Carla_Town01', 'Carla_Town02', 'Carla_Town03', 'Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15'],
                      nargs="+",
                      help='processed map name')
  parser.add_argument("--workers",
                      default=16,
                      help='multiprocessing workers num')
  opt = parser.parse_args()
  return opt

clip_config = {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": True,
  "do_convert_rgb": True,
  "do_normalize": True,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}
args = arg_parse()
processer = CLIPImageProcessor(**clip_config)

if __name__ == "__main__":
  def worker(traj_dir):
    traj_camera_list = []
    # 将该轨迹下5个摄像头文件夹里的图像拿出来，排序后存储
    # traj_camera_list[0] frontcamera里的图像文件，按文件名的顺序从小到大
    # traj_camera_list[1] leftcamera里的图像文件
    # 依次类推
    for idx,camera_name in enumerate(RGB_FOLDER):
      traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
    traj_frames = []
    # 默认所有相机中的图像文件一样多，都是55张
    # 遍历每一帧
    for idx in range(len(traj_camera_list[0])):
      batch = []
      # 将每一帧的5个视角图像路径打包成一个 batch， 长度为5
      for iid in range(len(RGB_FOLDER)):
          batch.append(traj_camera_list[iid][idx]) # size of batch (5, )
      # 将所有这样的batch装到traj_frames中
      traj_frames.append(batch) # size of traj_frames (55， 5)

    traj_imgs = []
    # 读取同一帧的张量
    for frame_imgs in traj_frames:
      # (5,) 路径 -> (5,) PIL 对象
      images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
      # (5,) PIL 对象 -> (5, 256, 256, 3) numpy 数组
      images = np.stack(images, axis=0)
      # 循环完55次后，traj_images 是长度55的list，每个元素是 (5, 256, 256, 3)的numpy数组
      traj_imgs.append(images)
    # (5*帧数， 256, 256, 3)
    imgs = np.array(traj_imgs).reshape(-1, 256, 256, 3)
    # 使用CLIP的图像预处理：resize 到 224 * 224、按CLIP的均值方差做normalize、转成 pixel_values 张量
    imgs = processer.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
    # 保存进文件 rgb_imgs.tensor
    torch.save(imgs, os.path.join(traj_dir, 'rgb_imgs.tensor'))

  # 串行处理各个地图
  for map_name in args.map_list:
    directory_path = os.path.join(args.root_dir, map_name)
    # 打印完整路径，如：/data/huangth/TravelUAV_dataset/extracted/Carla_Town01
    print(directory_path)
    traj_list = []
    for traj in tqdm.tqdm(os.listdir(directory_path)):
      traj_dir = os.path.join(directory_path, traj)
      # 以列表的形式存储 /Carla_Town01 文件夹下所有子文件夹（轨迹）的UUID
      traj_list.append(traj_dir)
    # 构建进程池
    with multiprocessing.Pool(args.workers) as p:
      # 调用worker函数处理轨迹，哪条轨迹先算完，就返回哪条轨迹
      # tqdm.tqdm 包一层进度条，显示 “这个地图的轨迹已经处理了多少条/一共多少条”
      # 强制等待所有任务执行完
      r = list(tqdm.tqdm(p.imap_unordered(worker, traj_list), total=len(traj_list)))
    print(directory_path, 'finished.')