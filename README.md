# Multi-Camera Anomaly Detection with CHAD Dataset

This project implements anomaly detection in multi-camera surveillance videos using the CHAD (Charlotte Anomaly Dataset).

## Features

- Multi-camera video processing
- Memory-based anomaly detection
- Streamlit GUI for visualization
- CHAD dataset support

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download CHAD dataset and place in `data/CHAD_Videos/` and `data/CHAD_Meta/`

3. Train the model:
   ```bash
   python train.py
   ```

4. Test the model:
   ```bash
   python test.py
   ```

5. Run the GUI:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `config.py`: Configuration settings
- `train.py`: Training script
- `test.py`: Testing script
- `app.py`: Streamlit GUI
- `datasets/`: Data loading
- `models/`: Model definitions
