"""数据加载"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_frames(video_path, clip_len=16, stride=8):
    """将视频切为固定长度片段"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    clips = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= clip_len:
            clips.append(np.array(frames[:clip_len]))
            frames = frames[stride:]  # 滑动窗口
    cap.release()
    return clips  # shape: (N, T, H, W, C)


def align_brightness(frame1, frame2):
    """简单亮度对齐，消除摄像头曝光差异"""
    lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2Lab).astype(float)
    lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2Lab).astype(float)
    lab1[:, :, 0] = lab1[:, :, 0] * (lab2[:, :, 0].mean() / (lab1[:, :, 0].mean() + 1e-5))
    return cv2.cvtColor(np.clip(lab1, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


class VideoClipDataset(Dataset):
    """上海科技视频异常检测训练/测试数据集"""

    def __init__(self, root_dir, split="training", clip_len=16, stride=8, num_cameras=1, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.num_cameras = num_cameras
        self.transform = transform

        videos_dir = os.path.join(root_dir, split, "videos")
        if not os.path.isdir(videos_dir):
            raise FileNotFoundError(f"找不到视频路径: {videos_dir}")

        self.video_files = sorted(
            [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.endswith(".avi")]
        )
        if len(self.video_files) == 0:
            raise RuntimeError(f"在 {videos_dir} 中没有找到视频文件")

        self.clip_index = []  # [(video_path, start_frame), ...]
        for vpath in self.video_files:
            cap = cv2.VideoCapture(vpath)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames < clip_len:
                continue
            n_clips = max(1, (n_frames - clip_len) // stride + 1)
            for i in range(n_clips):
                self.clip_index.append((vpath, i * stride))

        if len(self.clip_index) == 0:
            raise RuntimeError("没有找到可用的视频片段，请检查 clip_len/stride 设置")

    def __len__(self):
        return len(self.clip_index)

    def _read_clip(self, vpath, start):
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()

        if len(frames) < self.clip_len:
            # 重采样最后一帧填充
            while len(frames) < self.clip_len:
                frames.append(frames[-1].copy())

        clip_np = np.stack(frames, axis=0)  # [T, H, W, C]
        clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        if self.transform:
            clip = self.transform(clip)
        return clip

    def __getitem__(self, idx):
        if self.num_cameras == 1:
            vpath, start = self.clip_index[idx % len(self.clip_index)]
            return self._read_clip(vpath, start)

        # 生成 num_cameras 个不同视频流输入
        chosen = []
        base_idx = idx
        for c in range(self.num_cameras):
            vi = (base_idx + c) % len(self.clip_index)
            vpath, start = self.clip_index[vi]
            chosen.append(self._read_clip(vpath, start))

        # 形状 [num_cameras, C, T, H, W]
        return chosen
