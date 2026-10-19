# 🚦 YOLO11 Multi-Model Pipeline: Helmet & Motorcycle Detection

A computer vision pipeline that analyzes video of crowded traffic and detects four classes simultaneously: **Person, Motorcycle, Helmet, and Face**.

## 🚀 Motivation
No single model I found covered both vehicle detection and helmet detection. This project combines two YOLO11 models into one pipeline and merges their outputs into a single annotated video.

## 🛠️ How It Works
1. **Standard YOLO11 model (`yolo11n.pt`):** detects Persons (class 0) and Motorcycles (class 3).
2. **Pretrained YOLO11 helmet model (`helmet-11x.pt`):** detects Helmets and Faces.
3. **Pipeline:** The script reads the video frame by frame with OpenCV, runs both models on each frame, merges the bounding boxes onto one frame, and writes it to an output video, so the full video is never loaded into memory at once.

## ⚙️ Tech Stack
- **Language:** Python
- **Libraries:** Ultralytics (YOLO11), OpenCV, Hugging Face Hub
- **Environment:** Google Colab

## ▶️ Usage
Install dependencies:
```bash
pip install ultralytics opencv-python-headless huggingface_hub
```

Run:
```python
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Load models
model_path = hf_hub_download(repo_id="leeyunjai/yolo11-helmet", filename="helmet-11x.pt")
model_helmet = YOLO(model_path)
model_standard = YOLO("yolo11n.pt")  # persons and motorcycles

# 2. Set up video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# 3. Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Persons (0) and motorcycles (3)
    res_standard = model_standard.predict(frame, conf=0.5, classes=[0, 3], verbose=False)[0]
    annotated_frame = res_standard.plot()

    # Helmets and faces
    res_helmet = model_helmet.predict(frame, conf=0.5, iou=0.45, verbose=False)[0]
    annotated_frame = res_helmet.plot(img=annotated_frame)

    out.write(annotated_frame)

cap.release()
out.release()
print("Processing complete! Saved as output.mp4")
```

## 📹 Demo
A 19-second demo clip of a crowded street scene with many riders, pedestrians, and motion blur, showing all four classes detected with confidence scores.

> **Note:** No quantitative evaluation (mAP, FPS) has been done yet. The demo is qualitative only.

## 🙏 Credits
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- Helmet detection model: [leeyunjai/yolo11-helmet](https://huggingface.co/leeyunjai/yolo11-helmet)

## 🎯 Future Work
- Fine-tune the helmet model on region-specific data for better accuracy.
- Add quantitative evaluation (mAP, FPS).
- Optimize the pipeline for real-time inference on edge devices.

##Author : Mukarma Rehman 
-github: https://github.com/mukarma1
