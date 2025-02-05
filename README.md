# Real-Time Object Detection using YOLOv8 and OpenCV

## Overview
This project implements a real-time object detection system using YOLOv8 and OpenCV. It leverages the Ultralytics YOLOv8 model for detecting objects in live video feeds or static images. 

## Features
- **Real-time object detection** using YOLOv8.
- **Supports video streams, webcam feeds, and images**.
- **Customizable confidence threshold and NMS (Non-Maximum Suppression) settings**.
- **Uses OpenCV for efficient video processing**.
- **Scalable for different YOLOv8 models (nano, small, medium, large, xlarge)**.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

## Configuration
Modify parameters in `detect.py` to adjust settings like:
- `conf-thres`: Confidence threshold for detection.
- `iou-thres`: Non-Maximum Suppression (NMS) IoU threshold.
- `device`: Choose 'cpu' or 'cuda' for GPU acceleration.

## Output
- Detected objects will be displayed in real-time.
- Processed images/videos are saved in the `runs/detect/` directory.

## Dependencies
- `ultralytics`
- `opencv-python`
- `numpy`

## Future Improvements
- Add support for tracking objects.
- Implement an interface for tuning detection parameters.
- Optimize performance for mobile and embedded systems.



