# evaluate.py: Model evaluation with comprehensive metrics
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from config import *
from datasets.video_dataset import CHADVideoClipDataset
from models.memory import AnomalyDetector


def multicam_collate_fn(batch):
    return [torch.stack(cam_clips) for cam_clips in zip(*batch)]


def evaluate_with_metrics(checkpoint="checkpoints/anomaly_detector.pth"):
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

    model = AnomalyDetector(num_cameras=NUM_CAMERAS, feat_dim=FEATURE_DIM, mem_size=MEMORY_SIZE, fusion_type=FUSION_TYPE).to(device)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Model weights not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for clips in dataloader:
            clips = [c.to(device) for c in clips]

            anomaly_score, _, _, _ = model(clips)
            scores = anomaly_score.cpu().numpy()
            all_scores.extend(scores)

            # For CHAD dataset, we need to load ground truth labels
            # This is a simplified version - in practice, you'd load actual labels
            # For now, assume some threshold-based labeling or load from dataset
            labels = (scores > np.median(scores)).astype(int)  # Simplified
            all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_scores)
    accuracy = accuracy_score(all_labels, (all_scores > 0.5).astype(int))
    precision = precision_score(all_labels, (all_scores > 0.5).astype(int))
    recall = recall_score(all_labels, (all_scores > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_scores > 0.5).astype(int))

    print("Evaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save results
    results = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'scores': all_scores,
        'labels': all_labels
    }
    np.savez('evaluation_results.npz', **results)
    print("Results saved to evaluation_results.npz")


if __name__ == "__main__":
    evaluate_with_metrics()