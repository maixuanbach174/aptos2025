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
class DilatedResidualLayer(nn.Module):
    def __init__(self, d_in, d_out, kernel_size=3, dilation=1):
        super().__init__()
        self.conv_dilated = nn.Conv1d(d_in, d_out, kernel_size, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(d_out, d_out, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return out + x

class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_f_maps, num_f_maps, kernel_size=3, dilation=2**i)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, input_dim, num_classes):
        super().__init__()
        self.stage1 = SingleStageTCN(num_layers, num_f_maps, input_dim, num_classes)
        self.stages = nn.ModuleList([
            SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes)
            for _ in range(num_stages-1)
        ])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for stage in self.stages:
            out = stage(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

# -------- Dataset -------- #
def find_used_classes(feature_dir, label_dir, num_classes=35):
    """Scan all label files and return a sorted list of used class indices."""
    used = set()
    for f in os.listdir(feature_dir):
        if f.endswith('_feats.npy'):
            vid = f.replace('_feats.npy', '')
            label_path = os.path.join(label_dir, f"{vid}_labels.npy")
            if os.path.exists(label_path):
                labels = np.load(label_path)
                used.update(set(labels[labels >= 0].tolist()))
    return sorted(list(used))

def build_class_mapping(used_classes: List[int]) -> Dict[int, int]:
    """Map old class indices to new contiguous indices."""
    return {old: new for new, old in enumerate(used_classes)}

class SurgicalVideoDataset(Dataset):
    def __init__(self, feature_dir, label_dir, class_mapping=None):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.class_mapping = class_mapping
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
        # Remap labels if mapping is provided
        if self.class_mapping is not None:
            remapped = np.full_like(label, -100)
            for old, new in self.class_mapping.items():
                remapped[label == old] = new
            label = remapped
        label[label == -1] = -100
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    feats = pad_sequence(features, batch_first=True)
    labs = pad_sequence(labels, batch_first=True, padding_value=-100)
    return feats, labs

def compute_class_weights(dataset, num_classes):
    all_labels = []
    for _, labels in dataset:
        all_labels += labels[labels != -100].tolist()
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = [total / (counts.get(i, 1) + 1e-6) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.sum()

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
            feats = feats.permute(0, 2, 1).to(device)  # (B, D, T)
            labels = labels.to(device)                 # (B, T)
            
            outputs = model(feats)                     # (S, B, C, T)
            loss = 0
            for s in range(outputs.size(0)):
                loss += criterion(outputs[s].permute(0, 2, 1).reshape(-1, num_classes), 
                                labels.reshape(-1))
            
            # Get predictions from the final stage
            final_preds = outputs[-1].permute(0, 2, 1).argmax(dim=-1)  # (B, T)
            
            for pred_seq, label_seq in zip(final_preds, labels):
                mask = label_seq != -100
                all_preds.extend(pred_seq[mask].tolist())
                all_targets.extend(label_seq[mask].tolist())
            
            valid_frames = labels.ne(-100).sum().item()
            total_loss += loss.item() * valid_frames
            total_frames += valid_frames
    
    avg_loss = total_loss / total_frames
    return avg_loss, all_preds, all_targets

def train(feature_dir, label_dir, num_epochs=100, batch_size=4, learning_rate=1e-4, num_classes=35, 
          num_stages=4, num_layers=10, num_f_maps=64, model_save_path='MSTCN/models/mstcn_model.pth',
          plot_dir='MSTCN/plot', train_ratio=0.8):
    # -------- Training Setup -------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Find used classes and build mapping
    used_classes = find_used_classes(feature_dir, label_dir, num_classes)
    class_mapping = build_class_mapping(used_classes)
    new_num_classes = len(used_classes)
    print(f"Used classes: {used_classes}")
    print(f"Remapping {num_classes} classes to {new_num_classes} contiguous classes.")

    # Create dataset and split into train/test
    full_dataset = SurgicalVideoDataset(feature_dir, label_dir, class_mapping=class_mapping)
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # Compute class weights from training set only
    weights = compute_class_weights(train_dataset, new_num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

    model = MS_TCN(
        num_stages=num_stages,
        num_layers=num_layers,
        num_f_maps=num_f_maps,
        input_dim=2048,
        num_classes=new_num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -------- Training Loop -------- #
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_frames = 0
        
        for feats, labels in train_dataloader:
            feats = feats.permute(0, 2, 1).to(device)  # (B, D, T)
            labels = labels.to(device)                 # (B, T)

            outputs = model(feats)                     # (S, B, C, T)
            loss = 0
            for s in range(outputs.size(0)):
                loss += criterion(outputs[s].permute(0, 2, 1).reshape(-1, new_num_classes), 
                                labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_frames = labels.ne(-100).sum().item()
            total_train_loss += loss.item() * valid_frames
            total_train_frames += valid_frames

        avg_train_loss = total_train_loss / total_train_frames
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        avg_test_loss, test_preds, test_targets = evaluate_model(model, test_dataloader, criterion, device, new_num_classes)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with test loss: {best_test_loss:.4f}")
            
            # Plot confusion matrix for best model
            plot_confusion_matrix(test_preds, test_targets, new_num_classes, plot_dir)
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, plot_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss,
        'used_classes': used_classes
    }
    with open(os.path.join(plot_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    return model

def main():
    # Configuration
    feature_dir = 'dataset/features/resnet50'
    label_dir = 'dataset/labels'
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-3
    num_classes = 35  # original number, will be remapped
    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    model_save_path = 'MSTCN/models/mstcn_model.pth'
    plot_dir = 'MSTCN/plots'
    train_ratio = 0.8  # 80% training, 20% testing

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Train the model
    model = train(
        feature_dir=feature_dir,
        label_dir=label_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_classes=num_classes,
        num_stages=num_stages,
        num_layers=num_layers,
        num_f_maps=num_f_maps,
        model_save_path=model_save_path,
        plot_dir=plot_dir,
        train_ratio=train_ratio
    )

if __name__ == "__main__":
    main()
