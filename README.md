# Traffic Signal Violation Detection System

This project is a real-time AI-based Traffic Signal Violation Detection System. It uses YOLOv8 for vehicle detection, EasyOCR for license plate recognition, and OpenCV to monitor vehicles that cross a traffic signal during a red light. Violations are logged and stored in a MySQL database.

## Features

- **Vehicle Detection** using YOLOv8
- **Traffic Signal Monitoring** by analyzing light brightness
- **License Plate Recognition** using EasyOCR
- **Violation Logging** with vehicle type, confidence, timestamp, license plate
- **Violation Storage** in a MySQL database
- **Visual Display** with bounding boxes, text overlays, and real-time alerts

## Prerequisites

- Python 3.7+
- MySQL (Workbench or Server)
- Pre-trained YOLOv8 model (`yolov8m.pt`)
- Sample video file (`tr.mp4`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saanvireddy-05/traffic-signal-violation-detection.git
cd traffic-signal-violation-detection

### 2. Install Required Python Packages

```
bash
pip install -r requirements.txt


