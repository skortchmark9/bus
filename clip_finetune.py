import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ---- DATASET ----
class BusCropDataset(Dataset):
    def __init__(self, image_folder, label_csv, label_to_idx, add_unknown=True):
        self.image_folder = image_folder
        self.df = pd.read_csv(label_csv, header=None, names=["filename", "label"])
        self.label_to_idx = label_to_idx

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

        label_counts = self.df["label"].value_counts()
        print("Class distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row["filename"])
        img = preprocess(Image.open(img_path))
        label_idx = self.label_to_idx[row["label"]]
        return img, label_idx

# ---- MODEL ----
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# ---- TRAINING ----
def train_head(dataset, num_classes, epochs=30):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    head = ClassifierHead(512, num_classes).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                embeddings = model.encode_image(images).float()
            preds = head(embeddings)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

    return head

# ---- EVALUATION ----
def evaluate_head(model, head, dataset, label_to_idx):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    correct = defaultdict(int)
    total = defaultdict(int)

    model.eval()
    head.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model.encode_image(images).float()
            preds = head(embeddings).argmax(dim=1)

            for pred, true in zip(preds, labels):
                total[idx_to_label[true.item()]] += 1
                if pred.item() == true.item():
                    correct[idx_to_label[true.item()]] += 1

    print("\nPer-class accuracy:")
    for label in total:
        acc = 100 * correct[label] / total[label]
        print(f"{label:20s}: {acc:.2f}% ({correct[label]}/{total[label]})")

# ---- SAVING ----
def save_head_and_labels(head, label_to_idx, head_path="head.pth", labels_path="label_to_idx.json"):
    torch.save(head.state_dict(), head_path)
    with open(labels_path, "w") as f:
        json.dump(label_to_idx, f)
    print(f"Saved head to {head_path} and labels to {labels_path}")

def load_head_and_labels(head_path="head.pth", labels_path="label_to_idx.json"):
    with open(labels_path, "r") as f:
        label_to_idx = json.load(f)
    num_classes = len(label_to_idx)
    head = ClassifierHead(512, num_classes).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()
    return head, label_to_idx

# ---- MAIN (optional) ----
def main():
    label_csv = "data/crop_labels.csv"
    crop_folder = "data/bus_crops_labelled"

    df = pd.read_csv(label_csv, header=None, names=["filename", "label"])
    classes = sorted(df["label"].unique()) + ["unknown"]
    label_to_idx = {label: i for i, label in enumerate(classes)}
    dataset = BusCropDataset(crop_folder, label_csv, label_to_idx)

    head = train_head(dataset, num_classes=len(label_to_idx))
    evaluate_head(model, head, dataset, label_to_idx)
    save_head_and_labels(head, label_to_idx)

if __name__ == "__main__":
    main()
