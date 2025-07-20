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
```

### 2. Install Required Python Packages

```
bash
pip install -r requirements.txt
```

### 3. Set up MySQL Database
Open MySQL Workbench or any MySQL client and run the following commands:

```
bash
CREATE DATABASE traffic_violation;

USE traffic_violation;

CREATE TABLE violations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(38),
    vehicle_type VARCHAR(38),
    confidence FLOAT,
    license_plate VARCHAR(20),
    full_image_path TEXT,
    cropped_image_path TEXT
);
```

### 4. Download YOLOv8 Model
Download [`yolov8m.pt`](https://github.com/ultralytics/ultralytics/releases) from Ultralytics YOLOv8 Releases and place it inside the project directory.

### 5. Prepare Input Video
Place your traffic video file (e.g., tr.mp4) inside the project folder.

### 6. Run the Detection Script

```
bash
python traffic_violation_detection.py
```
## Project Structure

```
bash
traffic-signal-violation-detection/
├── yolov8m.pt                   # YOLOv8 model
├── tr.mp4                       # Sample traffic video
├── traffic_violation_detection.py
├── requirements.txt
├── violations/                 # Saved full-frame violation images
├── cropped/                    # Saved cropped vehicle images

```






