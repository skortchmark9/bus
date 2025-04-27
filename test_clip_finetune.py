from PIL import Image
import json
import torch
import torch.nn as nn

from clip_finetune import BusCropDataset, train_head, evaluate_head, save_head_and_labels, load_head_and_labels, ClassifierHead


def predict_crop(image_path, model, head_path="head.pth", labels_path="label_to_idx.json"):
    # Load label mapping
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Rebuild head
    num_classes = len(label_to_idx)
    head = ClassifierHead(512, num_classes).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()

    # Process image
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

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


# Load frozen CLIP model
import clip
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Predict
# prediction = predict_crop(
#     image_path="path/to/your_crop.jpg",
#     model=model,
#     head_path="head.pth",
#     labels_path="label_to_idx.json"
# )

# print(f"Prediction: {prediction}")
