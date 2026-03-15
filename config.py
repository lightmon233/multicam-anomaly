# Configuration for CHAD dataset anomaly detection
import os
import torch

# Dataset paths
DATA_ROOT = "data"
CHAD_VIDEOS_DIR = os.path.join(DATA_ROOT, "CHAD_Videos")
CHAD_META_DIR = os.path.join(DATA_ROOT, "CHAD_Meta")

# Model parameters
CLIP_LEN = 16
STRIDE = 8
NUM_CAMERAS = 4  # CHAD has 4 cameras
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# Image size
IMG_SIZE = (224, 224)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"