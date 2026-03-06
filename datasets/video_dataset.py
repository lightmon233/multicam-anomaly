"""数据加载"""
import cv2
import numpy as np

def extract_frames(video_path, clip_len=16, stride=8):
    """将视频切为固定长度片段"""
    cap = cv2.videoCapture(video_path)
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
            frames = frames[stride:] # 滑动窗口
    cap.release()
    return clips # shape: (N, T, H, W, C)

def align_brightness(frame1, frame2):
    """简单亮度对齐，消除摄像头曝光差异"""
    lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2Lab).astype(float)
    lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2Lab).astype(float)
    lab1[:, :, 0] = lab1[:, :, 0] * (lab2[:, :, 0].mean() / lab1[:, :, 0].mean())
    return cv2.cvtColor(np.clip(lab1, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)