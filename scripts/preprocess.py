#!/usr/bin/env python3
"""
Data preprocessing script for Argoverse 1 Motion Forecasting.

Converts raw Argoverse 1 data into preprocessed pickle files suitable for
the MGFNet model training pipeline.

Usage:
    python scripts/preprocess.py --data_dir /path/to/argoverse --output_dir data/processed
"""

import argparse
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_argoverse_track(track_data: Any, obs_len: int = 20) -> dict[str, np.ndarray]:
    """
    Parse a single agent track from Argoverse format.

    Args:
        track_data: track object from Argoverse dataset
        obs_len: observation horizon length (20 for Argoverse 1)

    Returns:
        dict with "track" [T, 2] and "timestamps" [T]
    """
    if hasattr(track_data, "states"):
        states = track_data.states
    elif isinstance(track_data, list):
        states = track_data
    else:
        raise ValueError(f"Unknown track format: {type(track_data)}")

    positions = []
    timestamps = []

    for state in states:
        if hasattr(state, "position"):
            positions.append([state.position[0], state.position[1]])
        elif hasattr(state, "x"):
            positions.append([state.x, state.y])
        elif isinstance(state, (list, np.ndarray)):
            positions.append([state[0], state[1]])

        if hasattr(state, "timestamp"):
            timestamps.append(state.timestamp)
        elif hasattr(state, "t"):
            timestamps.append(state.t)
        else:
            timestamps.append(0)

    return {
        "track": np.array(positions, dtype=np.float32),
        "timestamps": np.array(timestamps, dtype=np.float64),
    }


def process_scene(
    scene_path: str,
    output_dir: str,
    scene_id: str | None = None,
) -> None:
    """
    Process a single Argoverse scene file into preprocessed pickle.

    This is a template — actual implementation depends on the Argoverse
    API data format. Users should adapt this to their specific data.

    For the Argoverse 1 Motion Forecasting dataset format:
    - Each scene is a CSV with columns: TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME
    """
    import pandas as pd

    if scene_id is None:
        scene_id = os.path.splitext(os.path.basename(scene_path))[0]

    # Load raw data
    df = pd.read_csv(scene_path)

    # Get unique track IDs
    track_ids = df["TRACK_ID"].unique()

    # Identify target track (typically the one with OBJECT_TYPE == "AGENT")
    target_rows = df[df["OBJECT_TYPE"] == "AGENT"]
    if len(target_rows) == 0:
        print(f"  Skipping {scene_id}: no AGENT track found")
        return

    target_id = target_rows["TRACK_ID"].iloc[0]
    target_data = df[df["TRACK_ID"] == target_id].sort_values("TIMESTAMP")

    target_track = target_data[["X", "Y"]].values.astype(np.float32)
    target_timestamps = target_data["TIMESTAMP"].values.astype(np.float64)

    # Extract neighbor tracks
    neighbor_tracks_list = []
    other_ids = [tid for tid in track_ids if tid != target_id]

    for nid in other_ids:
        nbr_data = df[df["TRACK_ID"] == nid].sort_values("TIMESTAMP")
        if len(nbr_data) < 5:  # Skip very short tracks
            continue
        nbr_track = nbr_data[["X", "Y"]].values.astype(np.float32)
        nbr_ts = nbr_data["TIMESTAMP"].values.astype(np.float64)
        neighbor_tracks_list.append((nbr_track, nbr_ts))

    # Build scene dict
    scene_data = {
        "target_track": target_track,
        "target_timestamps": target_timestamps,
        "neighbor_tracks": neighbor_tracks_list,
        "scene_id": scene_id,
        "city_name": target_data["CITY_NAME"].iloc[0] if "CITY_NAME" in df.columns else "MIA",
    }

    # Save
    output_path = os.path.join(output_dir, f"{scene_id}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(scene_data, f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Argoverse 1 data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to raw Argoverse data (CSV directory)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for preprocessed files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Data split to process")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.split, "data")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(args.data_dir, args.split)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Looking for CSV files in the data directory...")
        data_dir = args.data_dir

    # Find CSV files
    csv_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ])

    if len(csv_files) == 0:
        # Try recursive
        csv_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        csv_files.sort()

    if len(csv_files) == 0:
        print(f"No CSV files found in {data_dir}")
        print("Creating empty output directory structure. Place your data here and re-run.")
        return

    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    print(f"Output directory: {output_dir}")

    for csv_path in tqdm(csv_files, desc="Processing scenes"):
        try:
            process_scene(csv_path, output_dir)
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")

    print(f"Preprocessing complete. Processed files saved to {output_dir}")


if __name__ == "__main__":
    main()
