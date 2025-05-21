#!/usr/bin/env python3
# extract_cnn_features.py

import os
import argparse
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.io import VideoReader
from PIL import Image

class VideoFrameDataset(Dataset):
    """
    Dataset that loads frames from a video using VideoReader in streaming mode,
    applies a transform, and returns the per-frame timestamp.
    """
    def __init__(self, video_path, transform):
        self.video_path = video_path
        self.transform = transform
        # Initialize video reader just to get metadata
        video = VideoReader(video_path)
        metadata = video.get_metadata()
        # Get FPS from metadata - it's a list, take the first value
        self.fps = float(metadata["video"]["fps"][0])
        # Get duration from metadata - it's a list, take the first value
        duration = float(metadata["video"]["duration"][0])
        self.T = int(duration * self.fps)
        # approximate uniform timestamps
        self.timestamps = np.arange(self.T, dtype=np.float32) / float(self.fps)

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        if idx >= self.T:
            raise IndexError("Index out of range")
        # Create a new VideoReader instance for each frame
        video = VideoReader(self.video_path)
        # Calculate timestamp for the frame
        timestamp = idx / self.fps
        # Seek to the frame
        video.seek(timestamp)
        # Get the next frame
        frame = next(video)
        # Convert to PIL Image - handle the frame data format correctly
        frame_data = frame["data"]  # This is a torch.Tensor
        # Convert to numpy and ensure correct shape (H, W, C)
        frame_np = frame_data.permute(1, 2, 0).numpy()
        img = Image.fromarray(frame_np)
        x = self.transform(img)            # [3,224,224]
        t = float(self.timestamps[idx])    # scalar
        return x, t

def build_backbone():
    """Load ResNet-50 pretrained and strip off final FC."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(model.children())[:-1]   # drop the last fc
    backbone = nn.Sequential(*modules)
    return backbone

def parse_args():
    p = argparse.ArgumentParser("Extract per-frame features with ResNet-50")
    p.add_argument("--data_path",   required=True,
                   help="Folder containing .mp4 videos")
    p.add_argument("--save_path",   required=True,
                   help="Output directory for .npy features & timestamps")
    p.add_argument("--part",   type=int, default=0,
                   help="This process index (for splitting workload)")
    p.add_argument("--total",  type=int, default=1,
                   help="Total number of parallel processes")
    p.add_argument("--gpu",    type=int, default=0,
                   help="GPU id")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for frame extraction")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Set pin_memory based on device type
    pin_memory = device.type == "cuda"  # Only use pin_memory for CUDA devices

    # find all .mp4 videos and split by part
    video_files = sorted(glob.glob(os.path.join(args.data_path, "*.mp4")))
    video_files = video_files[args.part::args.total]

    # prepare model & transform
    backbone = build_backbone().to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    for vid_path in tqdm(video_files, desc=f"Part {args.part}/{args.total}"):
        video_id = os.path.splitext(os.path.basename(vid_path))[0]
        feat_file = os.path.join(args.save_path, f"{video_id}_feats.npy")
        time_file = os.path.join(args.save_path, f"{video_id}_times.npy")

        # skip if already done
        if os.path.exists(feat_file) and os.path.exists(time_file):
            continue

        # build dataset & loader
        ds = VideoFrameDataset(vid_path, transform)
        loader = DataLoader(ds,
                            batch_size=args.batch_size,
                            num_workers=0,  # Use single worker to avoid serialization issues
                            pin_memory=pin_memory)  # Only use pin_memory for CUDA

        all_feats = []
        all_times = []

        with torch.no_grad():
            for batch in loader:
                imgs, times = batch              # imgs: [B,3,224,224], times: list of floats
                imgs = imgs.to(device)
                feats = backbone(imgs)           # [B,2048,1,1]
                feats = feats.view(feats.size(0), -1).cpu().numpy()  # [B,2048]
                all_feats.append(feats)
                all_times.extend(times.numpy() if isinstance(times, torch.Tensor) else times)

        # concatenate and save
        feats_arr = np.vstack(all_feats)           # [T,2048]
        times_arr = np.array(all_times, dtype=np.float32)  # [T]

        np.save(feat_file, feats_arr)
        np.save(time_file, times_arr)

    print("✅ Feature & timestamp extraction complete.")

if __name__ == "__main__":
    main()
