import torch
import torch.nn as nn

from models.encoder import MultiCameraEncoder


class MemoryModule(nn.Module):
    """存储正常行为原型的记忆槽"""

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


class AnomalyDetector(nn.Module):
    def __init__(self, num_cameras, feat_dim=512, mem_size=100):
        super().__init__()
        self.encoder = MultiCameraEncoder(num_cameras, feat_dim)
        # 注意力融合多摄像头特征
        self.fusion_attn = nn.Linear(feat_dim, 1)
        self.memory = MemoryModule(mem_size, feat_dim)
        # 解码器（用于重构，计算重构误差）
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, clips):
        feats = self.encoder(clips)  # list of [B, D]
        # 注意力融合：根据每个摄像头特征的重要性加权
        feat_stack = torch.stack(feats, dim=1)  # [B, num_cam, D]
        attn_w = torch.softmax(self.fusion_attn(feat_stack), dim=1)  # [B, num_cam, 1]
        fused = (attn_w * feat_stack).sum(dim=1)  # [B, D]

        retrieved, mem_attn = self.memory(fused)
        recon = self.decoder(retrieved)

        # 重构误差 = 异常得分
        anomaly_score = torch.norm(fused - recon, dim=1)  # [B]
        return anomaly_score, fused, recon, mem_attn