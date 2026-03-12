# test.py：模型推理与异常评分
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.video_dataset import VideoClipDataset
from models.memory import AnomalyDetector


def multicam_collate_fn(batch):
    if isinstance(batch[0], list):
        return list(zip(*batch))
    return torch.stack(batch)


def evaluate(
    data_root="data/ShanghaiTech",
    checkpoint="checkpoints/anomaly_detector.pth",
    num_cameras=2,
    clip_len=16,
    stride=8,
    batch_size=4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset(
        root_dir=data_root,
        split="testing",
        clip_len=clip_len,
        stride=stride,
        num_cameras=num_cameras,
    )

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=multicam_collate_fn,
    )

    model = AnomalyDetector(num_cameras=num_cameras).to(device)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"未找到模型权重: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_scores = []
    with torch.no_grad():
        for item in dataloader:
            if num_cameras == 1:
                clips = item.to(device)
                clips = [clips]
            else:
                clips = [torch.stack(cam, dim=0).to(device) for cam in item]

            anomaly_score, _, _, _ = model(clips)
            all_scores.extend(anomaly_score.cpu().numpy().tolist())

    out_path = "scores_test.npy"
    np.save(out_path, np.array(all_scores))
    print(f"测试完成，输出分数已保存：{out_path}")
    print(f"测试样本数量：{len(all_scores)}")


if __name__ == "__main__":
    evaluate()