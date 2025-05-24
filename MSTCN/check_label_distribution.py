#!/usr/bin/env python3
"""
check_label_distribution.py - Print label distribution for train and test splits
"""
import os
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, random_split
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Check label distribution in train/test splits')
    parser.add_argument('--feature_dir', type=str, default='dataset/features/resnet50', help='Feature directory')
    parser.add_argument('--label_dir', type=str, default='dataset/labels', help='Label directory')
    parser.add_argument('--num_classes', type=int, default=35, help='Number of classes')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    return parser.parse_args()

class SurgicalVideoDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.video_ids = []
        for f in os.listdir(feature_dir):
            if f.endswith('_feats.npy'):
                vid = f.replace('_feats.npy', '')
                label_path = os.path.join(label_dir, f"{vid}_labels.npy")
                if os.path.exists(label_path):
                    self.video_ids.append(vid)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feat = np.load(os.path.join(self.feature_dir, f"{vid}_feats.npy"))
        label = np.load(os.path.join(self.label_dir, f"{vid}_labels.npy"))
        label[label == -1] = -100
        return feat, label

def print_label_distribution(dataset, name, num_classes=35):
    all_labels = []
    for feats, labels in dataset:
        all_labels += labels[labels != -100].tolist()
    counts = Counter(all_labels)
    print(f"\nLabel distribution for {name} set:")
    for i in range(num_classes):
        print(f"Class {i}: {counts.get(i, 0)}")
    print(f"Total samples: {sum(counts.values())}\n")

def main():
    args = parse_args()
    full_dataset = SurgicalVideoDataset(args.feature_dir, args.label_dir)
    train_size = int(args.train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print_label_distribution(train_dataset, "train", args.num_classes)
    print_label_distribution(test_dataset, "test", args.num_classes)

if __name__ == "__main__":
    main() 