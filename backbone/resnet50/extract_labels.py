#!/usr/bin/env python3
"""
extract_labels.py

Given a directory of per-video feature files and timestamp files,
plus an annotation CSV, generate per-video label arrays.

Usage:
    python extract_labels.py \
        --feat_dir /path/to/feat_dir \
        --ann_csv /path/to/annotations.csv \
        [--out_dir /path/to/output_dir]

It reads all *_times.npy in feat_dir, matches each video_id,
creates *_labels.npy containing frame-level phase_id labels.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

def label_frames_for_video(video_id, timestamps, ann_df, default_label=-1):
    """
    Map each timestamp to its phase_id according to ann_df.
    ann_df must have columns: video_id, start, end, phase_id
    timestamps: np.array of shape [T]
    Returns labels: np.array [T]
    """
    # filter annotations for this video
    sub = ann_df[ann_df.video_id == video_id][['start', 'end', 'phase_id']].values
    T = len(timestamps)
    labels = np.full(T, default_label, dtype=np.int64)
    # for each frame, find matching segment
    for i, t in enumerate(timestamps):
        for start, end, pid in sub:
            if start <= t < end:
                labels[i] = int(pid)
                break
    return labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract per-frame phase labels from timestamps and annotations'
    )
    parser.add_argument('--feat_dir', required=True,
                        help='Directory containing *_times.npy and *_feats.npy files')
    parser.add_argument('--ann_csv', required=True,
                        help='Path to annotations CSV with columns video_id,start,end,phase_id')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for labels; default = feat_dir')
    parser.add_argument('--default_label', type=int, default=-1,
                        help='Label for frames outside any segment')
    args = parser.parse_args()

    feat_dir = args.feat_dir
    out_dir = args.out_dir or feat_dir
    os.makedirs(out_dir, exist_ok=True)

    # load annotation DataFrame
    ann_df = pd.read_csv(args.ann_csv)
    # ensure video_id column is string
    ann_df['video_id'] = ann_df['video_id'].astype(str)

    # process each _times.npy
    times_files = glob.glob(os.path.join(feat_dir, '*_times.npy'))
    if not times_files:
        print(f'No *_times.npy files found in {feat_dir}')
        exit(1)

    for times_path in times_files:
        video_id = os.path.basename(times_path).replace('_times.npy', '')
        labels_path = os.path.join(out_dir, f'{video_id}_labels.npy')
        if os.path.exists(labels_path):
            print(f'Skipping {video_id}, labels already exist')
            continue
        # load timestamps
        timestamps = np.load(times_path)
        # generate labels
        labels = label_frames_for_video(video_id, timestamps, ann_df, args.default_label)
        # save
        np.save(labels_path, labels)
        print(f'Saved labels for {video_id}: {labels_path}')
