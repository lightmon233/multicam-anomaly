import torch
import torch.nn as nn

from models.encoder import MultiCameraEncoder, SpatialTemporalAligner


class MemoryModule(nn.Module):
    """存储正常行为原型的记忆槽 - 基于开题报告的记忆增强模块"""

    def __init__(self, mem_size=100, feat_dim=512):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, feat_dim))

    def forward(self, query):
        # query: [B, D]
        # 计算与每个记忆槽的相似度
        mem_norm = nn.functional.normalize(self.memory, dim=1)  # [M, D]
        q_norm = nn.functional.normalize(query, dim=1)  # [B, D]
        attn = torch.softmax(q_norm @ mem_norm.T, dim=1)  # [B, M]
        # 加权读取记忆
        retrieved = attn @ self.memory  # [B, D]
        return retrieved, attn


class MultiSourceFusion(nn.Module):
    """多源融合模块 - 支持不同融合策略"""
    def __init__(self, num_cameras=4, feat_dim=512, fusion_type="attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_cameras = num_cameras
        self.feat_dim = feat_dim

        if fusion_type == "attention":
            # 注意力融合 - 自适应权重分配
            self.fusion_attn = nn.Linear(feat_dim, 1)
        elif fusion_type == "early":
            # 早期融合 - 特征拼接
            self.early_fusion = nn.Linear(feat_dim * num_cameras, feat_dim)
        elif fusion_type == "late":
            # 晚期融合 - 独立处理后加权
            self.late_weights = nn.Parameter(torch.ones(num_cameras))

    def forward(self, feats):
        # feats: list of [B, D] for each camera
        if self.fusion_type == "attention":
            # 注意力融合
            feat_stack = torch.stack(feats, dim=1)  # [B, num_cam, D]
            attn_w = torch.softmax(self.fusion_attn(feat_stack), dim=1)  # [B, num_cam, 1]
            fused = (attn_w * feat_stack).sum(dim=1)  # [B, D]
        elif self.fusion_type == "early":
            # 早期融合
            feat_concat = torch.cat(feats, dim=1)  # [B, num_cam * D]
            fused = self.early_fusion(feat_concat)  # [B, D]
        elif self.fusion_type == "late":
            # 晚期融合
            weights = torch.softmax(self.late_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, feats))
        else:
            # 默认平均融合
            fused = torch.stack(feats, dim=0).mean(dim=0)
        return fused


class AnomalyDetector(nn.Module):
    """多源视频异常检测器 - 符合开题报告要求"""
    def __init__(self, num_cameras=4, feat_dim=512, mem_size=100, fusion_type="attention"):
        super().__init__()
        self.encoder = MultiCameraEncoder(num_cameras, feat_dim)
        self.aligner = SpatialTemporalAligner(feat_dim)
        self.fusion = MultiSourceFusion(num_cameras, feat_dim, fusion_type)
        self.memory = MemoryModule(mem_size, feat_dim)
        # 解码器（用于重构，计算重构误差）
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, clips):
        feats = self.encoder(clips)  # list of [B, D]
        aligned_feats = self.aligner(feats)  # 时空对齐
        fused = self.fusion(aligned_feats)  # 多源融合

        retrieved, mem_attn = self.memory(fused)
        recon = self.decoder(retrieved)

        # 重构误差 = 异常得分
        anomaly_score = torch.norm(fused - recon, dim=1)  # [B]
        return anomaly_score, fused, recon, mem_attn