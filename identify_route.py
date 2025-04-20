import cv2
import numpy as np
import os
import glob
import easyocr
import matplotlib.pyplot as plt
import pytesseract
import torch
import clip
import cv2
from PIL import Image
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def match_clip_label(img, labels):
    # Convert OpenCV image to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    text_input = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        probs = similarity.softmax(dim=-1).cpu().numpy()[0]

    return sorted(zip(labels, probs), key=lambda x: -x[1])


def run_tesseract(img, psm=6):
    if type(img) is str:
        img = cv2.imread(img)

    config = f'--psm {psm} -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
    return pytesseract.image_to_string(img, config=config)


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


def extract_yellow_sign(img, margin=5):
    """
    Show the original image, yellow mask, and detected crop region with bounding box and projection line.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.dilate(mask, np.ones((3, 15), np.uint8), iterations=1)

    coords = cv2.findNonZero(mask)
    if coords is None:
        print("No yellow pixels found.")
        return

    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x + w + 2*margin, img.shape[1])
    y2 = min(y + h + 2*margin, img.shape[0])
    cropped = img[y:y2, x:x2]

    debug = False
    if not debug:
        return cropped


    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].add_patch(plt.Rectangle((x, y), x2-x, y2-y, edgecolor='lime', facecolor='none', linewidth=2))
    axs[0].set_title("Original + Yellow Bounding Box")
    axs[0].axis('off')

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Yellow Mask (Dilated)")
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Final Cropped Sign Region")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    return cropped




def show(i):
    plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return i

def run_ocr_on_image(img, min_conf=0.0):
    if type(img) is str:
        img = cv2.imread(img)
    ocr_reader = easyocr.Reader(['en'])
    results = ocr_reader.readtext(img)
    print(results)
    return [(text, conf, bbox) for (bbox, text, conf) in results if conf >= min_conf]

def show_ocr_results(img, ocr_results):
    display_img = img.copy()
    for text, conf, bbox in ocr_results:
        print(text, conf, bbox)
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(display_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(display_img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def try_ocr_on_masked_images(output_dir=output_path):
    masked_images = glob.glob(os.path.join(output_dir, '*.jpg'))
    for image_path in masked_images:
        img = cv2.imread(image_path)
        ocr_yellow_characters(img)
        # ocr_results = run_ocr_on_image(img)
        # show_ocr_results(img, ocr_results)


def extract_char_regions(img, min_area=20, pad_px=2, show_debug=False):
    """
    Extract yellow blobs using connected component stats (better for black bg).
    Returns cropped RGB patches.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    patches = []
    h_img, w_img = img.shape[:2]

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        x1 = max(x - pad_px, 0)
        y1 = max(y - pad_px, 0)
        x2 = min(x + w + pad_px, w_img)
        y2 = min(y + h + pad_px, h_img)

        char_crop = img[y1:y2, x1:x2]
        patches.append(((x1, y1, x2, y2), char_crop))

    # Sort left to right by x1
    patches.sort(key=lambda tup: tup[0][0])

    if show_debug:
        debug_img = img.copy()
        for (x1, y1, x2, y2), _ in patches:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Yellow Char Boxes (Connected Components)")
        plt.axis('off')
        plt.show()

    return [p[1] for p in patches]  # return just images


def ocr_yellow_characters(img):
    chars = []
    candidates = [
        "This is an M1 bus", "This is an M2 bus", "This is an M3 bus",
        "This is an M7 bus", "This is an M10 bus", "I can't tell what bus this is",
        "It's too small to tell"
    ]
    results = match_clip_label(img, candidates)
    print(results[0])
    show(img)
    return

    patches = extract_char_regions(img, show_debug=True)
    for i, patch in enumerate(patches):
        candidates = [
            "This is an M1 bus", "This is an M2 bus", "This is an M3 bus",
            "This is an M7 bus", "This is an M10 bus", "I can't tell what bus this is"
        ]
        result = match_clip_label(patch, candidates)
        print(result)
        confidence = result[0][1]
        result = result[0][0]
        print(f"Detected: {result}")
        chars.append(result)
        plt.subplot(1, len(patches), i+1)
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.title(f"'{result}' ({confidence:.2f})")
        plt.axis('off')
    plt.show()
    return ''.join(chars)
