import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class MultiCameraEncoder(nn.Module):
    def __init__(self, num_cameras=4, feat_dim=512):
        super().__init__()
        # 每个摄像头一个独立的编码器分支（不共享权重）
        self.branches = nn.ModuleList([
            self._build_branch() for _ in range(num_cameras)
        ])
        self.feat_dim = feat_dim

    def _build_branch(self):
        backbone = r2plus1d_18(pretrained=True)
        # 去掉分类头，只保留特征提取部分
        return nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, clips):
        # clips: list of [B, C, T, H, W], 每个摄像头一个
        feats = []
        for i, clip in enumerate(clips):
            f = self.branches[i](clip)  # [B, 512, 1, 1, 1]
            f = f.flatten(1)            # [B, 512]
            feats.append(f)
        return feats  # list of [B, 512]


class SpatialTemporalAligner(nn.Module):
    """时空对齐模块 - 基于开题报告要求"""
    def __init__(self, feat_dim=512):
        super().__init__()
        # 时间对齐 - 基于光流相似度
        self.temporal_align = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
        # 空间对齐 - 基于特征匹配
        self.spatial_align = nn.MultiheadAttention(feat_dim, num_heads=8)

    def forward(self, feats):
        # feats: list of [B, D] for each camera
        aligned_feats = []
        for i, feat in enumerate(feats):
            # 简化的时空对齐 - 这里可以扩展为更复杂的对齐算法
            aligned_feat = feat  # 暂时保持不变
            aligned_feats.append(aligned_feat)
        return aligned_feats