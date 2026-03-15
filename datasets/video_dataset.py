"""Data loading for CHAD dataset"""
import os
import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CHADVideoClipDataset(Dataset):
    """CHAD dataset for multi-camera anomaly detection"""

    def __init__(self, root_dir, split="train_split_1", clip_len=16, stride=8, num_cameras=4, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.num_cameras = num_cameras
        self.transform = transform

        # Paths
        self.videos_dir = os.path.join(root_dir, "CHAD_Videos")
        self.meta_dir = os.path.join(root_dir, "CHAD_Meta")
        self.annotations_dir = os.path.join(self.meta_dir, "annotations")
        self.labels_dir = os.path.join(self.meta_dir, "anomaly_labels")
        self.splits_dir = os.path.join(self.meta_dir, "splits")

        # Load split
        split_file = os.path.join(self.splits_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.video_names = [line.strip() for line in f.readlines()]

        # Group by video number (remove camera and anomaly flag)
        self.video_groups = {}
        for name in self.video_names:
            # name format: cam_video_anomaly.mp4, but without extension in splits?
            # Assuming splits contain names without .mp4
            if name.endswith('.mp4'):
                name = name[:-4]
            parts = name.split('_')
            video_num = parts[1]
            if video_num not in self.video_groups:
                self.video_groups[video_num] = []
            self.video_groups[video_num].append(name)

        # For each video group, create clips
        self.clip_index = []  # [(video_num, start_frame), ...]
        for video_num, names in self.video_groups.items():
            # Assume all cameras have same frame count, use first one
            first_name = names[0] + '.mp4'
            vpath = os.path.join(self.videos_dir, first_name)
            if not os.path.exists(vpath):
                continue
            cap = cv2.VideoCapture(vpath)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames < clip_len:
                continue
            n_clips = max(1, (n_frames - clip_len) // stride + 1)
            for i in range(n_clips):
                self.clip_index.append((video_num, i * stride))

        if len(self.clip_index) == 0:
            raise RuntimeError("No valid clips found")

    def __len__(self):
        return len(self.clip_index)

    def _read_clip(self, video_num, camera, start):
        name = f"{camera}_{video_num}_0.mp4"  # Assume normal for now, but need to check
        # Actually, need to find the correct name
        group = self.video_groups[video_num]
        for n in group:
            if n.startswith(f"{camera}_{video_num}"):
                name = n + '.mp4'
                break
        vpath = os.path.join(self.videos_dir, name)
        if not os.path.exists(vpath):
            # Create blank clip if missing
            return torch.zeros(3, self.clip_len, 224, 224)

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
            while len(frames) < self.clip_len:
                frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3)))

        clip_np = np.stack(frames, axis=0)  # [T, H, W, C]
        clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        if self.transform:
            clip = self.transform(clip)
        return clip

    def __getitem__(self, idx):
        video_num, start = self.clip_index[idx]
        clips = []
        for cam in range(1, self.num_cameras + 1):
            clip = self._read_clip(video_num, cam, start)
            clips.append(clip)
        # Return list of clips [num_cameras, C, T, H, W]
        return clips
