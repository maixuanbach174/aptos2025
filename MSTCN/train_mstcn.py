#!/usr/bin/env python3
"""
train_mstcn.py

Train a simple MS-TCN on pre-extracted per-frame features for frame-level phase recognition,
and provide a CLI for training and inference.

Usage:
  # Train
  python train_mstcn.py --feat_dir /path/to/features \
                        --label_dir /path/to/labels \
                       --out_model mstcn_phase.pth \
                       --epochs 10 --bs 4 --lr 1e-3

  # Inference on one video
  python train_mstcn.py --predict case_0001_feats.npy \
                       --model mstcn_phase.pth
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ── Dataset ────────────────────────────────────────────────────────────
class FeatureSeqDataset(Dataset):
    def __init__(self, feat_dir, label_dir, seq_len=512, stride=256, augment=True):
        self.examples = []
        self.augment = augment
        # load all videos
        feat_files = glob.glob(os.path.join(feat_dir, "*_feats.npy"))
        print(f"Found {len(feat_files)} feature files in {feat_dir}")
        
        if len(feat_files) == 0:
            raise ValueError(f"No feature files found in {feat_dir}")
            
        for feat_path in feat_files:
            vid = os.path.basename(feat_path).replace("_feats.npy","")
            label_path = os.path.join(label_dir, f"{vid}_labels.npy")
            
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {vid}, skipping...")
                continue
                
            feats = np.load(feat_path)                      # [T, D]
            labels = np.load(label_path)                    # [T]
            
            print(f"Loading {vid}: features shape {feats.shape}, labels shape {labels.shape}")
            
            T, D = feats.shape
            for start in range(0, T, stride):
                end = min(start + seq_len, T)
                fs = feats[start:end]
                ls = labels[start:end]
                if len(fs) < seq_len:
                    pad = seq_len - len(fs)
                    fs = np.concatenate([fs, np.tile(fs[-1:], (pad,1))], axis=0)
                    ls = np.concatenate([ls, np.tile(ls[-1:], pad)], axis=0)
                self.examples.append((fs, ls))
                
        if len(self.examples) == 0:
            raise ValueError("No valid examples were loaded from the dataset")
            
        print(f"[Dataset] {len(self.examples)} chunks loaded from {len(feat_files)} videos")
        print(f"First chunk shapes - Features: {self.examples[0][0].shape}, Labels: {self.examples[0][1].shape}")

    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        f, l = self.examples[i]
        if self.augment:
            # Random noise augmentation
            if np.random.random() < 0.5:
                noise = np.random.normal(0, 0.01, f.shape)
                f = f + noise
            # Random scaling
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                f = f * scale
        return torch.from_numpy(f).float(), torch.from_numpy(l).long()

# ── Model ──────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class MSTCN(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, layers=4):
        super().__init__()
        self.input_conv = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(hidden)
        self.input_relu = nn.ReLU(inplace=True)
        
        # Residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden, hidden, dilation=2**i)
            for i in range(layers)
        ])
        
        self.classifier = nn.Conv1d(hidden, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B,T,D] → [B,D,T]
        x = x.permute(0,2,1)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        
        for block in self.res_blocks:
            x = block(x)
            x = self.dropout(x)
            
        x = self.classifier(x)
        # → [B,T,C]
        return x.permute(0,2,1)

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, labs in dataloader:
            feats, labs = feats.to(device), labs.to(device)
            logits = model(feats)
            # Ensure tensors are contiguous before reshaping
            logits = logits.contiguous()
            preds = logits.reshape(-1, num_classes).argmax(-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labs.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, conf_matrix

def plot_confusion_matrix(conf_matrix, num_classes, epoch, save_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()

# ── Train & Predict ────────────────────────────────────────────────────
def train(feat_dir, label_dir, out_model, epochs, bs, lr, seq_len, stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create dataset and split into train/val
    full_dataset = FeatureSeqDataset(feat_dir, label_dir, seq_len, stride, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # infer dims
    sample = np.load(glob.glob(os.path.join(feat_dir, "*_feats.npy"))[0])
    in_dim = sample.shape[1]
    labels = np.load(glob.glob(os.path.join(label_dir, "*_labels.npy"))[0])
    num_classes = 35  # APTOS has 35 classes
    
    # Calculate class weights
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))  # Add small epsilon to avoid division by zero
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    model = MSTCN(in_dim, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Use label smoothing
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Create directory for saving plots
    save_dir = os.path.dirname(out_model)
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    for ep in range(1, epochs+1):
        # Training
        model.train()
        train_loss = 0
        for feats, labs in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]"):
            feats, labs = feats.to(device), labs.to(device)
            logits = model(feats)
            # Ensure tensors are contiguous before reshaping
            logits = logits.contiguous()
            labs = labs.contiguous()
            # Use reshape instead of view
            loss = loss_fn(logits.reshape(-1, num_classes), labs.reshape(-1))
            opt.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        val_acc, conf_matrix = evaluate(model, val_loader, device, num_classes)
        
        print(f"Epoch {ep} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save confusion matrix
        plot_confusion_matrix(conf_matrix, num_classes, ep, save_dir)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_model)
            print(f"[Saved] Best model with val accuracy: {val_acc:.4f}")

    print(f"[Training Complete] Best validation accuracy: {best_val_acc:.4f}")
    return model

def predict(model_path, feat_file, label_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
    # load model
    sample = np.load(glob.glob(os.path.dirname(feat_file)+"/*_feats.npy")[0])
    in_dim = sample.shape[1]
    labels = np.load(label_file)
    num_classes = len(np.unique(labels))
    model = MSTCN(in_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    feats = torch.from_numpy(np.load(feat_file)).unsqueeze(0).to(device)  # [1,T,D]
    with torch.no_grad():
        out = model(feats)[0]             # [T,C]
        preds = out.argmax(-1).cpu().numpy()
    print(f"Pred shape: {preds.shape}")
    print("First 20 preds:", preds[:20])

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir",  help="dir with *_feats.npy/_labels.npy", required=True)
    p.add_argument("--label_dir", help="dir with *_labels.npy", required=True)
    p.add_argument("--out_model", help="where to save weights", default="mstcn_phase.pth")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--bs",        type=int, default=4)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--seq_len",   type=int, default=512)
    p.add_argument("--stride",    type=int, default=256)
    p.add_argument("--predict",   help="feat_file to run inference on")
    p.add_argument("--label_file", help="label file for prediction")
    args = p.parse_args()

    if args.predict:
        if not args.label_file:
            p.error("--label_file is required when using --predict")
        predict(args.out_model, args.predict, args.label_file)
    else:
        train(args.feat_dir, args.label_dir, args.out_model,
              args.epochs, args.bs, args.lr,
              args.seq_len, args.stride)
