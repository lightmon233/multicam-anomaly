# train.py core logic for CHAD dataset
import os
import random

import torch
from torch.utils.data import DataLoader

from config import *
from datasets.video_dataset import CHADVideoClipDataset
from models.detector import compute_loss
from models.memory import AnomalyDetector


def multicam_collate_fn(batch):
    # batch is list of lists: [clips for cam1, clips for cam2, ...]
    return [torch.stack(cam_clips) for cam_clips in zip(*batch)]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    device = torch.device(DEVICE)

    set_seed(42)

    dataset = CHADVideoClipDataset(
        root_dir=DATA_ROOT,
        split="train_split_1",
        clip_len=CLIP_LEN,
        stride=STRIDE,
        num_cameras=NUM_CAMERAS,
    )

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=multicam_collate_fn,
    )

    model = AnomalyDetector(num_cameras=NUM_CAMERAS, feat_dim=FEATURE_DIM, mem_size=MEMORY_SIZE, fusion_type=FUSION_TYPE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_samples = 0.0, 0

        for batch_idx, clips in enumerate(dataloader, start=1):
            clips = [c.to(device) for c in clips]  # list of [B, C, T, H, W] for each cam

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
                    f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f} | Avg {total_loss / total_samples:.4f}"
                )

        print(f"Epoch {epoch} completed, average loss {total_loss / max(1,total_samples):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "anomaly_detector.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved: {ckpt_path}")


if __name__ == "__main__":
    train()
