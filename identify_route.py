import cv2
import numpy as np
import os
import glob

path = 'data/nyc-bus.v3i.yolov8'
output_path = 'data/masked_images'


def select_images_with_bboxes(path:str):
    image_list = glob.glob(os.path.join(path, '*/images', '*.jpg'))
    images_with_bboxes = []
    for image_name in image_list:
        path = image_name.replace('/images/', '/labels/')
        path = path.replace('.jpg', '.txt')
        label_file = path
        with open(label_file, 'r') as file:
            contents = file.read()
            if contents:
                boxes = parse_yolo_boxes(contents)
                images_with_bboxes.append((image_name, boxes))
    return images_with_bboxes


def parse_yolo_boxes(contents:str):
    yolo_list = contents.splitlines()
    parsed = []
    for line in yolo_list:
        tokens = line.strip().split()
        # class_id = int(tokens[0])
        coords = list(map(float, tokens[1:]))
        parsed.append(coords)
    return parsed

def mask_image(image_path, coords):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(coords) == 4:
        # Rectangle: YOLO format (cx, cy, w, h)
        cx, cy, bw, bh = coords
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    elif len(coords) % 2 == 0:
        # Polygon
        points = np.array([
            [int(x * w), int(y * h)] for x, y in zip(coords[::2], coords[1::2])
        ], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return cv2.bitwise_and(img, img, mask=mask)

def mask_images_with_bboxes(images_with_bboxes:list, output_dir=output_path):
    os.makedirs(output_dir, exist_ok=True)
    for image_path, bboxes in images_with_bboxes:
        for i, bbox in enumerate(bboxes):
            # Mask the image with the bounding box
            img = mask_image(image_path, bbox)
            image_name = os.path.basename(image_path) + '_' + str(i) + '.jpg'

            # Save masked image
            new_image_path = os.path.join(output_dir, 'masked_' + image_name)
            cv2.imwrite(new_image_path, img)
