import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CAM_FOLDER = Path("site/output_one_cam")


def load_data():
    # **1. Point this at your camera’s merged folder**

    # **2. Gather feature vectors**
    features = []
    bus_ids  = []

    for track_dir in CAM_FOLDER.iterdir():
        info_path = track_dir / "info.json"
        if not info_path.exists():
            continue

        data = json.load(info_path.open())
        bboxes = data["bboxes"]       # list of [x1,y1,x2,y2]
        timestamps = data["timestamps"]  # list of timestamps
        n_frames = len(bboxes)
        if n_frames < 2:
            continue

        # Normalize coords to [0,1] by image size if you know it; 
        # here we assume a fixed size (e.g. 1920×1080), or skip norm for raw pixels:
        frame_size = (352, 240)
        W, H = frame_size

        # Entry & exit centroids
        def centroid(bb):
            x1, y1, x2, y2 = bb
            return ((x1+x2)/2 / W, (y1+y2)/2 / H)

        x_in, y_in   = centroid(bboxes[0])
        x_out, y_out = centroid(bboxes[-1])

        # Dwell time (in frames; you could convert to seconds if you know fps)
        timestamps = [datetime.strptime(ts, "%Y%m%dT%H%M%S") for ts in timestamps]

        grid_feats = grid_frame_counts(bboxes, frame_size, grid_size=4)
        features.append([x_in, y_in, x_out, y_out] + list(grid_feats))
        bus_ids.append(track_dir.name)

    features = np.array(features)
    return features, bus_ids

def grid_frame_counts(
    bboxes: list[list[float]],
    frame_size: tuple[int,int],
    grid_size: int
) -> np.ndarray:
    """
    Build a grid histogram of where the bus appeared in each frame.
    
    Args:
      bboxes: list of [x1, y1, x2, y2] for each frame
      frame_size: (W, H) in pixels
      grid_size: number of cells per row/col (G)
    
    Returns:
      1D np.array of length G*G, counts per cell (row-major).
    """
    W, H = frame_size
    G = grid_size
    counts = np.zeros((G, G), dtype=int)
    
    for x1, y1, x2, y2 in bboxes:
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        # map [0,1] to cell idx [0..G-1]
        j = min(int(cx * G), G - 1)
        i = min(int(cy * G), G - 1)
        counts[i, j] += 1
    
    return counts.ravel()

def mark_outliers(
    clusters: pd.Series,
    min_size: int = 2,
    outlier_label: int = -1
) -> pd.Series:
    """
    Keep clusters of size >= min_size, but reassign any cluster
    smaller than that to `outlier_label`.
    """
    counts = clusters.value_counts()
    small = counts[counts < min_size].index
    cleaned = clusters.copy()
    cleaned[cleaned.isin(small)] = outlier_label
    return cleaned

def kmeans(features, bus_ids, K=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=K, random_state=42).fit(X_scaled)

    labels = km.labels_

    # 1. Zip them together
    clusters = pd.Series(labels, index=bus_ids, name='cluster')
    clusters = clusters.sort_index(key=lambda idx: idx.map(lambda s: int(s.split('___')[-1])))

    clusters_clean = mark_outliers(clusters, min_size=2, outlier_label=-1)

    for bus_id, lbl in clusters_clean.items():
        print(f"Bus {bus_id} → cluster {lbl}")

    return clusters_clean


def copy(results):
    for bus_id, label in results.items():
        src = CAM_FOLDER / bus_id
        if not src.exists():
            print(f"Skipping missing folder: {bus_id}")
            continue

        dest = Path('data/clustered') / f"cluster_{label}" / bus_id
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy entire folder tree
        shutil.copytree(src, dest)


def sweep_k(features, k_min=2, k_max=10):
    """
    Run KMeans for K in [k_min..k_max], return lists of Ks, inertias, and silhouette scores.
    
    features: np.ndarray of shape (n_samples, n_features)
    """
    Ks = list(range(k_min, k_max + 1))
    inertias = []
    silhs = []
    for k in Ks:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        inertias.append(km.inertia_)
        silhs.append(silhouette_score(features, km.labels_))
    return Ks, inertias, silhs

def plot_sweep(Ks, inertias, silhs):
    """
    Plot inertia (elbow) and silhouette curves on a single figure.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(Ks, inertias, marker='o', label='Inertia')
    ax1.set_xlabel('Number of clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(Ks, silhs, marker='s', linestyle='--', label='Silhouette')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend(loc='upper right')
    plt.title('Elbow & Silhouette Analysis')
    plt.tight_layout()
    plt.show()

def best_k_by_silhouette(Ks, silhs):
    """
    Return the K corresponding to the maximum silhouette score.
    """
    idx = int(np.argmax(silhs))
    return Ks[idx]
