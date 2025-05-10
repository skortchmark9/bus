import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from resnet import RoutePredictor as ResnetRoutePredictor
from test_clip_finetune import RoutePredictor as ClipRoutePredictor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


def load_all_filenames(crop_folder):
    return sorted([p.name for p in crop_folder.glob("*.jpg")])


def load_labels_with_unknown(filenames, label_csv):
    labels_df = pd.read_csv(label_csv, names=["filename", "true_label"])
    label_dict = dict(zip(labels_df["filename"], labels_df["true_label"]))

    records = []
    for fname in filenames:
        label = label_dict.get(fname, "unknown")
        records.append({"filename": fname, "true_label": label})

    return pd.DataFrame(records)


def run_predictions(df, crop_folder, clip_model, resnet_model):
    clip_preds, resnet_preds = [], []

    for fname in tqdm(df["filename"], desc="Running predictions"):
        img_path = crop_folder / fname
        img = cv2.imread(str(img_path))

        if img is None:
            clip_preds.append("error")
            resnet_preds.append("error")
            continue

        clip_label, _ = clip_model.predict(img)
        resnet_label, _ = resnet_model.predict(img)

        clip_preds.append(clip_label)
        resnet_preds.append(resnet_label)

    df["clip_pred"] = clip_preds
    df["resnet_pred"] = resnet_preds
    return df


def determine_winner(row):
    if row["clip_pred"] == row["true_label"] and row["resnet_pred"] != row["true_label"]:
        return "clip"
    elif row["resnet_pred"] == row["true_label"] and row["clip_pred"] != row["true_label"]:
        return "resnet"
    elif row["resnet_pred"] == row["clip_pred"] == row["true_label"]:
        return "tie"
    else:
        return "neither"


def summarize_model_comparison(df):
    from collections import Counter

    # Overall accuracy
    clip_acc = (df["clip_pred"] == df["true_label"]).mean()
    resnet_acc = (df["resnet_pred"] == df["true_label"]).mean()

    # Winner counts
    winner_counts = df["winner"].value_counts()

    # Per-class accuracy
    per_class = df.groupby("true_label").apply(
        lambda x: pd.Series({
            "n": len(x),
            "clip_acc": (x["clip_pred"] == x["true_label"]).mean(),
            "resnet_acc": (x["resnet_pred"] == x["true_label"]).mean(),
            "clip_wins": (x["winner"] == "clip").sum(),
            "resnet_wins": (x["winner"] == "resnet").sum(),
            "ties": (x["winner"] == "tie").sum(),
            "neither": (x["winner"] == "neither").sum(),
        })
    ).reset_index()

    print("=== Overall Accuracy ===")
    print(f"CLIP:   {clip_acc:.2%}")
    print(f"ResNet: {resnet_acc:.2%}")
    print()
    print("=== Winner Counts ===")
    print(winner_counts.to_string())
    print()
    print("=== Per-Class Accuracy ===")
    print(per_class.sort_values("n", ascending=False))


def evaluate_resnet(df):
    y_true = df["true_label"]
    y_pred = df["resnet_pred"]

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=3))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    cm_df = pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique()))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ResNet Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cm_df



def run_comparison(crop_dir="data/examples/4_27_1500/crops_labelled",
                   output_csv="model_comparison_results.csv"):

    crop_folder = Path(crop_dir)
    label_csv = crop_folder / "crop_labels.csv"
    all_filenames = load_all_filenames(crop_folder)
    clip_predictor = ClipRoutePredictor()
    resnet_predictor = ResnetRoutePredictor()
    df = load_labels_with_unknown(all_filenames, label_csv)
    df = run_predictions(df, crop_folder, clip_predictor, resnet_predictor)
    df["winner"] = df.apply(determine_winner, axis=1)

    df.to_csv(output_csv, index=False)
    print(f"[✓] Wrote results to: {output_csv}")
    return df


if __name__ == "__main__":
    run_comparison()
