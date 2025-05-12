import json
import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from sklearn.cluster import KMeans, DBSCAN
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA


def load_info(track_dir: Path) -> dict:
    """
    Load the info.json from a given track directory.
    """
    with track_dir.joinpath("info.json").open("r") as f:
        return json.load(f)


def normalized_centers(bboxes: list, frame_width: int, frame_height: int) -> np.ndarray:
    """
    Convert list of bboxes to an array of normalized (cx, cy, size).
    """
    centers = []
    for x1, y1, x2, y2 in bboxes:
        cx = (x1 + x2) / 2 / frame_width
        cy = (y1 + y2) / 2 / frame_height
        size = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
        centers.append((cx, cy, size))
    return np.array(centers)


def build_trajectory_dataframe(
    camera_folder: Path,
    frame_width: int,
    frame_height: int
) -> pd.DataFrame:
    """
    Build a long-format DataFrame with columns ['bus_id', 'time', 'cx', 'cy', 'size']
    for each trajectory in the camera folder.
    """
    rows = []
    for track_dir in camera_folder.iterdir():
        info_path = track_dir.joinpath("info.json")
        if not info_path.exists():
            continue

        info = load_info(track_dir)
        traj = normalized_centers(info['bboxes'], frame_width, frame_height)
        timestamps = info["timestamps"]  # list of timestamps
        timestamps = [datetime.strptime(ts, "%Y%m%dT%H%M%S") for ts in timestamps]

        # skip too-short tracks
        if traj.shape[0] < 3:
            continue

        bus_id = track_dir.name
        for t, (cx, cy, size) in zip(timestamps, traj):
            rows.append({
                'bus_id': bus_id,
                'time': t,
                'cx': cx,
                'cy': cy,
                'size': size
            })

    return pd.DataFrame(rows)


def extract_tsfresh_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract tsfresh features from the long-form DataFrame.
    Returns a DataFrame indexed by 'bus_id'.
    """
    features_df = extract_features(
        df,
        column_id='bus_id',
        column_sort='time',
        default_fc_parameters=None,
        show_warnings=False
    )
    # Impute any NaNs left after feature extraction
    impute(features_df)

    # 1) drop any columns that are constant (zero variance)
    nunique = features_df.nunique()
    nonconstant = nunique[nunique > 1].index
    # 2) also drop any columns still containing NaNs (just in case)
    cleaned = features_df[nonconstant].dropna(axis=1, how="any")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cleaned.values)
    return pd.DataFrame(scaled, index=cleaned.index, columns=cleaned.columns)


def cluster_kmeans(
    features_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42
) -> pd.Series:
    """
    Run KMeans on the feature DataFrame and return a Series of cluster labels.
    Indexed by bus_id.
    """
    X = features_df.values
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    series = pd.Series(labels, index=features_df.index, name='cluster')

    # Sort by the integer after '___' in bus_id
    def sort_key(idx: pd.Index) -> pd.Index:
        return idx.map(lambda s: int(s.split('___')[-1]))

    clusters = series.sort_index(key=sort_key)

    counts = clusters.value_counts().sort_index()
    print("Cluster sizes:")
    print(counts.to_string())
    min_size = 2
    small = counts[counts < min_size]
    if not small.empty:
        print("\nWarning: these clusters have fewer than", min_size, "members:")
        for lbl, cnt in small.items():
            print(f"  cluster_{lbl} → {cnt} member(s)")

    return clusters

def cluster_dbscan(features_df: pd.DataFrame):
    n_components = 10
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(features_df.values)

    X = features_df.values
    eps = 1.0
    db = DBSCAN(eps=eps, min_samples=2, metric='cosine')
    labels = db.fit_predict(X_reduced)
    print(eps)

    series = pd.Series(labels, index=features_df.index, name='cluster')

    # Sort by the integer after '___' in bus_id
    def sort_key(idx: pd.Index) -> pd.Index:
        return idx.map(lambda s: int(s.split('___')[-1]))

    clusters = series.sort_index(key=sort_key)

    counts = clusters.value_counts().sort_index()
    print("Cluster sizes:")
    print(counts.to_string())
    min_size = 2
    small = counts[counts < min_size]
    if not small.empty:
        print("\nWarning: these clusters have fewer than", min_size, "members:")
        for lbl, cnt in small.items():
            print(f"  cluster_{lbl} → {cnt} member(s)")

    return clusters


def copy_cluster_folders(
    clusters: pd.Series,
    camera_folder: Path,
    output_root: Path
) -> None:
    """
    Copy each bus's folder into subdirectories under output_root/cluster_<label>/bus_id.

    Parameters:
    - clusters: Series mapping bus_id to cluster label (index bus_id).
    - camera_folder: Path where original bus_id subfolders reside.
    - output_root: Path where 'cluster_<label>' folders will be created.
    """
    import shutil

    for bus_id, label in clusters.items():
        src = camera_folder / bus_id
        if not src.exists() or not src.is_dir():
            continue

        dest = output_root / f"cluster_{label}" / bus_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        # If destination exists, skip or overwrite? Here we skip.
        if dest.exists():
            continue

        shutil.copytree(src, dest)

def sweep_k(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> tuple[list[int], list[float], list[float]]:
    """
    Run KMeans for K in [k_min..k_max], return lists of Ks, inertias, and silhouette scores.
    Assumes features are already cleaned and scaled if desired.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Ks = list(range(k_min, k_max + 1))
    inertias = []
    silhs = []
    X = features_df.values
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
        labels = km.labels_
        silhs.append(silhouette_score(X, labels))
    return Ks, inertias, silhs


def plot_sweep(
    Ks: list[int],
    inertias: list[float],
    silhs: list[float]
) -> None:
    """
    Plot inertia (elbow) and silhouette curves on a single figure.
    """
    import matplotlib.pyplot as plt

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
