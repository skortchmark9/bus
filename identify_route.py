import cv2
import numpy as np
import os
import glob

path = 'data/nyc-bus.v3i.yolov8'
output_path = 'data/masked_images'

def mask_images(path:str):
    image_list = os.listdir(path)
    for image_name in image_list:
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path)
        # Apply mask here

        # Save masked image
        new_image_path = os.path.join(path, 'masked_' + image_name)
        cv2.imwrite(new_image_path, img)


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
    tokens = contents.strip().split()
    boxes = []
    # Chunk every 5 tokens (YOLO: class x y w h)
    for i in range(0, len(tokens), 5):
        if i + 5 <= len(tokens):
            boxes.append(tokens[i:i+5])
    return boxes

def mask_image(image_path, yolo_boxes):
    # Load image
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Create black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Parse and un-normalize YOLO boxes
    for box in yolo_boxes:
        _, x_center, y_center, box_w, box_h = map(float, box)

        # Convert to pixel coordinates
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # Draw white rectangle in the mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # Apply mask
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img

def mask_images_with_bboxes(output_dir=output_path, images_with_bboxes:list):
    os.makedirs(os.path.join(output_dir, 'masked_images'), exist_ok=True)
    for image_path, bboxes in images_with_bboxes:
        img = mask_image(image_path, bboxes)

        image_name = os.path.basename(image_path)

        # Save masked image
        new_image_path = os.path.join(output_dir, 'masked_images', 'masked_' + image_name)
        cv2.imwrite(new_image_path, img)
