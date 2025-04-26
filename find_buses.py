from ultralytics import YOLO
import shutil
import os
from pathlib import Path
import glob
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

def is_mta_blue(img, min_frac=0.01):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Narrower, darker blue range — tuned for MTA navy
    lower_blue = np.array([100, 100, 50])   # hue 100–130, high sat, moderate val
    upper_blue = np.array([130, 255, 200])



    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    return blue_ratio > min_frac

import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np
from collections import deque

def has_yellow_on_dark(image, yellow_dark_ratio_thresh=0.2, max_region_size=5000, brightness_std_thresh=10):
    """
    Region-growing based detection of yellow-on-dark, with brightness texture check.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 1. Yellow mask
    lower_yellow = np.array([22, 50, 220])
    upper_yellow = np.array([28, 100, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. Dark mask
    value_channel = hsv[..., 2]
    dark_mask = (value_channel < 60).astype(np.uint8)

    # 3. Initialize
    h, w = yellow_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    # Seed from yellow pixels
    yellow_coords = np.argwhere(yellow_mask > 0)
    for y, x in yellow_coords:
        queue.append((y, x))
        visited[y, x] = True

    yellow_count = 0
    dark_count = 0
    total_region_size = 0

    region_pixels = []

    # 4. Region growing
    while queue:
        y, x = queue.popleft()

        if yellow_mask[y, x] > 0 or dark_mask[y, x] > 0:
            region_pixels.append(value_channel[y, x])

        if yellow_mask[y, x] > 0:
            yellow_count += 1
        elif dark_mask[y, x] > 0:
            dark_count += 1

        total_region_size += 1
        if total_region_size > max_region_size:
            break

        # 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if yellow_mask[ny, nx] > 0 or dark_mask[ny, nx] > 0:
                        queue.append((ny, nx))
                        visited[ny, nx] = True

    if yellow_count + dark_count == 0:
        return False

    # 5. Calculate yellow-dark ratio
    ratio = yellow_count / (yellow_count + dark_count)

    # 6. Calculate brightness stddev
    if region_pixels:
        brightness_std = np.std(region_pixels)
    else:
        brightness_std = 0

    return (ratio >= yellow_dark_ratio_thresh) and (brightness_std >= brightness_std_thresh)





def process_bus_crop(image, target_size=256):
    # Resize and pad to target size
    return ImageOps.pad(image, (target_size, target_size), color=(0, 0, 0))

def downselect_crops(input_folder, output_folder):
    image_paths = sorted(Path(input_folder).rglob("*.jpg"))
    os.makedirs(output_folder, exist_ok=True)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR, fix to RGB

        if has_yellow_on_dark(img_rgb):
            filename = os.path.basename(image_path)
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, img)



def extract_buses_from_dir(
    image_dir, 
    output_dir="data/bus", 
    model_path="yolov8n.pt", 
    conf_thresh=0.3, 
    limit=None
):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # find the last image in the output directory to start the count
    count = len(sorted(Path(output_dir).rglob("*.jpg")))
    print("Starting count:", count)

    image_paths = sorted(Path(image_dir).rglob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            result = model(img, conf=conf_thresh, classes=[5])[0]
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
                expanded_crop = np.array(process_bus_crop(Image.fromarray(crop)))
                # if has_yellow_on_dark(expanded_crop):
                cv2.imwrite(out_path, expanded_crop)
                count += 1

        # if found_bus:
        #     # Copy the image to the output directory
        #     output_img_path = os.path.join(output_dir, img_path.name)
        #     shutil.copy(img_path, output_img_path)
        #     count += 1


def copy_annotated_buses_from_cvat(path="data/nyc-bus.v3i.yolov8", out="tmp"):
    """Copy annotated bus images from CVAT, split by train/test/valid to a new flat directory
    with a json file containing bounding boxes."""
    os.makedirs(out)
    image_paths = glob.glob(path + '/*/images/*.jpg')
    labels = {}
    for image_path in image_paths:
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        name = os.path.basename(image_path)
        with open(label_path, 'r') as f:
            contents = f.read()
            if contents:
                shutil.copy(image_path, out)
                labels[name] = contents

    with open(out + '/labels.json', 'w') as f:
        json.dump(labels, f, indent=4)
