#!/usr/bin/env python3
"""
train_mstcn.py - Multi-Stage Temporal Convolutional Network for surgical phase recognition
"""

import os
# Set OpenMP environment variable to handle duplicate library initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from typing import Dict, List

# -------- MS-TCN Model -------- #
class TemporalCNN(nn.Module):
    def __init__(self, input_dim=2048, num_classes=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1)
        )

    def forward(self, x):  # x: (B, T, D)
        x = x.permute(0, 2, 1)  # → (B, D, T)
        x = self.net(x)         # → (B, C, T)
        return x.permute(0, 2, 1)  # → (B, T, C)

# -------- Dataset -------- #
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
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    feats = pad_sequence(features, batch_first=True)
    labs = pad_sequence(labels, batch_first=True, padding_value=-100)
    return feats, labs

def plot_training_curves(train_losses, test_losses, save_dir):
    """Plot training and testing loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(predictions, targets, num_classes, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, 
                cmap="Blues", 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes), 
                annot=False, 
                fmt=".2f")
    
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Phase")
    plt.ylabel("True Phase")
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def evaluate_model(model, dataloader, criterion, device, num_classes):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    total_frames = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)  # (B, T, D)
            labels = labels.to(device)  # (B, T)
            outputs = model(feats)  # (B, T, C)
            loss = criterion(outputs.reshape(-1, num_classes), labels.reshape(-1))
            preds = outputs.argmax(dim=-1)  # (B, T)
            for pred_seq, label_seq in zip(preds, labels):
                mask = label_seq != -100
                all_preds.extend(pred_seq[mask].tolist())
                all_targets.extend(label_seq[mask].tolist())
            valid_frames = labels.ne(-100).sum().item()
            total_loss += loss.item() * valid_frames
            total_frames += valid_frames
    avg_loss = total_loss / total_frames
    return avg_loss, all_preds, all_targets

def predict_and_save(model, dataset, device, save_path, batch_size=4):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    all_preds = []
    all_vids = dataset.video_ids
    with torch.no_grad():
        idx = 0
        for feats, labels in dataloader:
            feats = feats.to(device)
            outputs = model(feats)  # (B, T, C)
            preds = outputs.argmax(dim=-1).cpu().numpy()  # (B, T)
            for i in range(preds.shape[0]):
                vid = all_vids[idx]
                np.save(f"{save_path}/{vid}_preds.npy", preds[i][:len(labels[i])])
                idx += 1
    print(f"Saved predictions for {idx} videos to {save_path}")

def train(feature_dir, label_dir, num_epochs=100, batch_size=4, learning_rate=1e-4, num_classes=35, 
          model_save_path='MSTCN/models/mstcn_model.pth', plot_dir='MSTCN/plot', train_ratio=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    full_dataset = SurgicalVideoDataset(feature_dir, label_dir)
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    model = TemporalCNN(input_dim=2048, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_frames = 0
        for feats, labels in train_dataloader:
            feats = feats.to(device)
            labels = labels.to(device)
            outputs = model(feats)
            loss = criterion(outputs.reshape(-1, num_classes), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            valid_frames = labels.ne(-100).sum().item()
            total_train_loss += loss.item() * valid_frames
            total_train_frames += valid_frames
        avg_train_loss = total_train_loss / total_train_frames
        train_losses.append(avg_train_loss)
        avg_test_loss, test_preds, test_targets = evaluate_model(model, test_dataloader, criterion, device, num_classes)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with test loss: {best_test_loss:.4f}")
            plot_confusion_matrix(test_preds, test_targets, num_classes, plot_dir)
    plot_training_curves(train_losses, test_losses, plot_dir)
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss
    }
    with open(os.path.join(plot_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    return model, full_dataset

def main():
    feature_dir = 'dataset/features/resnet50'
    label_dir = 'dataset/labels'
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-3
    num_classes = 35
    model_save_path = 'MSTCN/models/mstcn_model.pth'
    plot_dir = 'MSTCN/plots'
    pred_dir = 'MSTCN/preds'
    train_ratio = 0.8
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    model, full_dataset = train(
        feature_dir=feature_dir,
        label_dir=label_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_classes=num_classes,
        model_save_path=model_save_path,
        plot_dir=plot_dir,
        train_ratio=train_ratio
    )
    predict_and_save(model, full_dataset, device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'), save_path=pred_dir, batch_size=batch_size)

if __name__ == "__main__":
    main()
