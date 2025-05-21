import numpy as np
from pathlib import Path

# point this to one of your videos
video_id = "case_1886"  
feat_dir = Path("./dataset/features/resnet50")  # where you saved the .npy files
label_dir = Path("./dataset/labels/")

# load
feats  = np.load(feat_dir / f"{video_id}_feats.npy")   # shape [T, D]
times  = np.load(feat_dir / f"{video_id}_times.npy")   # shape [T]
labels = np.load(label_dir / f"{video_id}_labels.npy")  # shape [T]

print(f"feats shape : {feats[0:feats.shape[0]: 100].shape}")
# print(f"times shape : {times.shape}")
print(f"labels shape: {labels[0:labels.shape[0]: 100].shape}")