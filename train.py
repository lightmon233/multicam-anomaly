# train.py 核心逻辑
import os
import random

import torch
from torch.utils.data import DataLoader

from datasets.video_dataset import VideoClipDataset
from models.detector import compute_loss
from models.memory import AnomalyDetector


def multicam_collate_fn(batch):
    if isinstance(batch[0], list):
        return list(zip(*batch))
    return torch.stack(batch)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_root="data/ShanghaiTech",
    epochs=10,
    batch_size=4,
    num_cameras=2,
    clip_len=16,
    stride=8,
    lr=1e-4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    dataset = VideoClipDataset(
        root_dir=data_root,
        split="training",
        clip_len=clip_len,
        stride=stride,
        num_cameras=num_cameras,
    )

    # DataLoader 输出: list[cam] (每个cam: [B, C, T, H, W])
    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=multicam_collate_fn,
    )

    model = AnomalyDetector(num_cameras=num_cameras).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0

        for batch_idx, item in enumerate(dataloader, start=1):
            if num_cameras == 1:
                clips = item.to(device)
                clips = [clips]
            else:
                clips = [torch.stack(cam, dim=0).to(device) for cam in item]

            scores, fused, recon, mem_attn = model(clips)
            loss = compute_loss(fused, recon, mem_attn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = clips[0].shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f} | Avg {total_loss / total_samples:.4f}"
                )

        print(f"Epoch {epoch} 完成，平均损失 {total_loss / max(1,total_samples):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "anomaly_detector.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"模型已保存: {ckpt_path}")


if __name__ == "__main__":
    train(epochs=5, batch_size=4, num_cameras=2, clip_len=16, stride=8)
