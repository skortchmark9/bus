import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from collections import defaultdict
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- AUGMENTATION ----
def get_augmentation_pipeline():
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8),
        T.RandomApply([
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
        ], p=0.3),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean/std
    ])

# ---- DATASET ----
class BusCropDataset(Dataset):
    def __init__(self, image_folder, label_csv, label_to_idx, transform=None, add_unknown=True):
        self.image_folder = image_folder
        self.df = pd.read_csv(label_csv, header=None, names=["filename", "label"])
        self.label_to_idx = label_to_idx
        self.transform = transform

        if add_unknown:
            all_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
            labeled_images = set(self.df["filename"])
            unlabeled_images = list(set(all_images) - labeled_images)

            n_known = len(self.df)
            random.shuffle(unlabeled_images)
            selected_unknowns = unlabeled_images[:n_known]

            unknown_df = pd.DataFrame({
                "filename": selected_unknowns,
                "label": "unknown"
            })

            self.df = pd.concat([self.df, unknown_df], ignore_index=True)
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row["filename"])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_idx = self.label_to_idx[row["label"]]
        return img, label_idx

# ---- MODEL ----
def load_resnet50(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_balanced_sampler(dataset):
    label_counts = dataset.df["label"].value_counts().to_dict()
    weights = [1.0 / label_counts[row["label"]] for _, row in dataset.df.iterrows()]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)

# ---- TRAINING ----
def train_resnet(model, dataset, epochs=30):
    sampler = create_balanced_sampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

    return model

# ---- EVALUATION ----
def evaluate_resnet(model, dataset, label_to_idx):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    correct = defaultdict(int)
    total = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)

            for pred, true in zip(preds, labels):
                total[idx_to_label[true.item()]] += 1
                if pred.item() == true.item():
                    correct[idx_to_label[true.item()]] += 1

    print("\nPer-class accuracy:")
    for label in total:
        acc = 100 * correct[label] / total[label]
        print(f"{label:20s}: {acc:.2f}% ({correct[label]}/{total[label]})")

# ---- SAVING ----
def save_model_and_labels(model, label_to_idx, model_path="resnet.pth", labels_path="resnet_label_to_idx.json"):
    torch.save(model.state_dict(), model_path)
    with open(labels_path, "w") as f:
        json.dump(label_to_idx, f)
    print(f"Saved model to {model_path} and labels to {labels_path}")

def load_model_and_labels(model_path="resnet.pth", labels_path="resnet_label_to_idx.json"):
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    num_classes = len(label_to_idx)
    model = load_resnet50(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, label_to_idx

# ---- INFERENCE FUNCTIONS ----
def predict_crop(img_arr, model, label_to_idx):
    img = Image.fromarray(img_arr).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0).to(device)

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)

    label = idx_to_label[pred_idx.item()]
    confidence = prob.item()
    return label, confidence

class RoutePredictor:
    def __init__(self, model_path="resnet.pth", labels_path="resnet_label_to_idx.json"):
        self.model, self.label_to_idx = load_model_and_labels(model_path, labels_path)
        self.model = self.model.to(device)

    def predict_file(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.predict(img)

    def predict(self, img):
        if isinstance(img, Image.Image):
            img_arr = np.array(img)
        else:
            img_arr = img
        return predict_crop(img_arr, self.model, self.label_to_idx)

# ---- MAIN ----
def main():
    label_csv = "data/crop_labels.csv"
    crop_folder = "data/bus_crops_labelled"

    df = pd.read_csv(label_csv, header=None, names=["filename", "label"])
    classes = sorted(df["label"].unique()) + ["unknown"]
    label_to_idx = {label: i for i, label in enumerate(classes)}

    transform = get_augmentation_pipeline()
    dataset = BusCropDataset(crop_folder, label_csv, label_to_idx, transform=transform)

    model = load_resnet50(num_classes=len(label_to_idx))
    model = train_resnet(model, dataset, epochs=30)
    evaluate_resnet(model, dataset, label_to_idx)
    save_model_and_labels(model, label_to_idx)

if __name__ == "__main__":
    main()
