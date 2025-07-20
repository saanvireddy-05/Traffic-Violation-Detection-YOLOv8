import cv2
import os
from ultralytics import YOLO
import numpy as np
import time
import easyocr
import mysql.connector

# Setup folders
os.makedirs("violations", exist_ok=True)
os.makedirs("cropped", exist_ok=True)

# MySQL DB connection
db = mysql.connector.connect(
    host="localhost",
    user="root",  # change if needed
    password="Saanvireddy@05",  # change this!
    database="traffic_violation"
)
cursor = db.cursor()

# Load model and OCR
model = YOLO("yolov8m.pt")
coco = model.model.names
reader = easyocr.Reader(['en'])

TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
cooldown_dict = {}              # license plate: last violation time
violated_plates = set()         # memory of plates that violated

# Polygons
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

# Helper functions
def is_region_light(image, polygon, brightness_threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [polygon], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.mean(roi, mask=mask)[0] > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale, text_color, bg_color, border_color, thickness=2, padding=5):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + baseline + padding), bg_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + baseline + padding), border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

def recognize_license_plate(img):
    results = reader.readtext(img)
    for result in results:
        text = result[1]
        if 6 <= len(text) <= 12 and all(c.isalnum() or c in '- ' for c in text):
            return text.upper().replace(" ", "")
    return "Unknown"

# Video
cap = cv2.VideoCapture("tr.mp4")
violation_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("✅ Video processing complete.")
        break

    frame = cv2.resize(frame, (1100, 700))
    red_on = is_region_light(frame, RedLight)

    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    results = model.predict(frame, conf=0.75)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            label = coco[int(cls)]
            if label in TargetLabels:
                x1, y1, x2, y2 = map(int, box)
                cropped = frame[y1:y2, x1:x2]
                license_plate = recognize_license_plate(cropped)

                # Violation logic
                is_violation_now = red_on and (
                    cv2.pointPolygonTest(ROI, (x1, y1), False) >= 0 or
                    cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0
                )

                if is_violation_now:
                    violated_plates.add(license_plate)

                is_already_violator = license_plate in violated_plates
                box_color = (0, 0, 255) if is_violation_now or is_already_violator else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                draw_text_with_background(
                    frame, f"{label.capitalize()} {(conf*100):.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), box_color
                )

                if is_violation_now:
                    now = time.time()
                    if license_plate not in cooldown_dict or now - cooldown_dict[license_plate] > 30:
                        cooldown_dict[license_plate] = now
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        full_path = f"violations/violation_{timestamp}.jpg"
                        crop_path = f"cropped/{label}_{timestamp}.jpg"
                        cv2.imwrite(full_path, frame)
                        cv2.imwrite(crop_path, cropped)

                        # Insert into MySQL
                        insert_query = """
                        INSERT INTO violations (timestamp, vehicle_type, confidence, license_plate, full_image_path, cropped_image_path)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (timestamp, label, float(f"{conf:.2f}"), license_plate, full_path, crop_path))
                        db.commit()

                        violation_count += 1
                        draw_text_with_background(
                            frame, f"🚨 Violation: {label}, Plate: {license_plate}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0), (0, 0, 255)
                        )

    draw_text_with_background(
        frame, f"Violations: {violation_count}", (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), (255, 255, 0)
    )

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
