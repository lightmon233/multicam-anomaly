# test.py: Model inference and anomaly scoring for CHAD dataset
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from datasets.video_dataset import CHADVideoClipDataset
from models.memory import AnomalyDetector


def multicam_collate_fn(batch):
    return [torch.stack(cam_clips) for cam_clips in zip(*batch)]


def evaluate(checkpoint="checkpoints/anomaly_detector.pth"):
    device = torch.device(DEVICE)

    dataset = CHADVideoClipDataset(
        root_dir=DATA_ROOT,
        split="test_split_1",
        clip_len=CLIP_LEN,
        stride=STRIDE,
        num_cameras=NUM_CAMERAS,
    )

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=multicam_collate_fn,
    )

    model = AnomalyDetector(num_cameras=NUM_CAMERAS).to(device)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Model weights not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for clips in dataloader:
            clips = [c.to(device) for c in clips]

            anomaly_score, _, _, _ = model(clips)
            all_scores.extend(anomaly_score.cpu().numpy().tolist())

    out_path = "scores_test.npy"
    np.save(out_path, np.array(all_scores))
    print(f"Testing completed, scores saved: {out_path}")
    print(f"Number of test samples: {len(all_scores)}")


if __name__ == "__main__":
    evaluate()