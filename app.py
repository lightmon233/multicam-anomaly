import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import *
from datasets.video_dataset import CHADVideoClipDataset
from models.memory import AnomalyDetector

st.title("CHAD Multi-Camera Anomaly Detection")

# Load model
@st.cache_resource
def load_model():
    device = torch.device(DEVICE)
    model = AnomalyDetector(num_cameras=NUM_CAMERAS).to(device)
    checkpoint = "checkpoints/anomaly_detector.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()
    return model, device

model, device = load_model()

# Select video
st.header("Select Test Video")
dataset = CHADVideoClipDataset(
    root_dir=DATA_ROOT,
    split="test_split_1",
    clip_len=CLIP_LEN,
    stride=STRIDE,
    num_cameras=NUM_CAMERAS,
)

video_options = list(dataset.video_groups.keys())
selected_video = st.selectbox("Choose a video number:", video_options)

if selected_video:
    st.write(f"Selected video: {selected_video}")

    # Get clips for this video
    clips_indices = [i for i, (vid, _) in enumerate(dataset.clip_index) if vid == selected_video]

    if clips_indices:
        clip_idx = st.slider("Select clip index:", 0, len(clips_indices)-1, 0)
        actual_idx = clips_indices[clip_idx]

        # Load the clip
        clips = dataset[actual_idx]  # list of tensors [C, T, H, W] for each cam

        # Display frames
        st.header("Video Frames")
        cols = st.columns(NUM_CAMERAS)
        for cam in range(NUM_CAMERAS):
            with cols[cam]:
                st.write(f"Camera {cam+1}")
                # Show middle frame
                frame = clips[cam][:, CLIP_LEN//2, :, :].permute(1, 2, 0).numpy()
                frame = (frame * 255).astype(np.uint8)
                st.image(frame, use_column_width=True)

        # Run detection
        if st.button("Detect Anomalies"):
            with torch.no_grad():
                clips_tensor = [c.unsqueeze(0).to(device) for c in clips]  # add batch dim
                anomaly_score, _, _, _ = model(clips_tensor)
                score = anomaly_score.item()

            st.header("Anomaly Score")
            st.write(f"Anomaly Score: {score:.4f}")

            # Visualize score
            fig, ax = plt.subplots()
            ax.bar(['Normal', 'Anomaly'], [1-score, score])
            ax.set_ylabel('Probability')
            st.pyplot(fig)

            if score > 0.5:
                st.error("Anomaly Detected!")
            else:
                st.success("Normal Behavior")
    else:
        st.write("No clips available for this video")