from PIL import Image
import json
import torch
import torch.nn as nn
import clip

from clip_finetune import BusCropDataset, train_head, evaluate_head, save_head_and_labels, load_head_and_labels, ClassifierHead

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def predict_crop(img_arr, model, head, label_to_idx):
    img = Image.fromarray(img_arr)  # <<< convert numpy -> PIL

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Process image
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(img).float()
        logits = head(embedding)
        probs = torch.softmax(logits, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)


    label = idx_to_label[pred_idx.item()]
    confidence = prob.item()
    print('label:', label)
    print('confidence:', confidence)

    return label, confidence

class RoutePredictor:
    def __init__(self, head_path="head.pth", labels_path="label_to_idx.json"):
        self.head_path = head_path
        self.labels_path = labels_path
        self.model = model
        self.head, self.label_to_idx = load_head_and_labels(head_path, labels_path)

    def predict_file(self, image_path):
        img = Image.open(image_path)
        return self.predict(img)

    def predict(self, img):
        return predict_crop(img, self.model, self.head, self.label_to_idx)
