from ultralytics import YOLO
import shutil
import os
from pathlib import Path

import cv2
import numpy as np

def is_mta_blue(img, min_frac=0.01):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Narrower, darker blue range — tuned for MTA navy
    lower_blue = np.array([100, 100, 50])   # hue 100–130, high sat, moderate val
    upper_blue = np.array([130, 255, 200])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    return blue_ratio > min_frac



def extract_buses_from_dir(
    image_dir, 
    output_dir="data/bus", 
    model_path="yolov8n.pt", 
    conf_thresh=0.3, 
    limit=None
):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(Path(image_dir).rglob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            result = model(img, conf=conf_thresh)[0]
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        if img is None:
            continue

        found_bus = False
        for i, box in enumerate(result.boxes.data):
            cls = int(box[5])
            if cls != 5:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            crop = img[y1:y2, x1:x2]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue  # skip degenerate crops

            if is_mta_blue(crop):  # 🔵 apply color filter to the crop
                out_path = os.path.join(output_dir, f"{count:06d}.jpg")
                cv2.imwrite(out_path, img)
                count += 1

        if found_bus:
            # Copy the image to the output directory
            output_img_path = os.path.join(output_dir, img_path.name)
            shutil.copy(img_path, output_img_path)
            count += 1

