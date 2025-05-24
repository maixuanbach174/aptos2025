import os
# Set OpenMP environment variable to handle duplicate library initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from your training script
from MSTCN.train_mstcn import MS_TCN, SurgicalVideoDataset, pad_collate_fn

# ---------- Metrics ---------- #
def frame_accuracy(preds, labels, ignore_index=-100):
    valid = labels != ignore_index
    correct = (preds == labels) & valid
    return correct.sum().item() / valid.sum().item()

def compute_edit_distance(pred, label):
    def compress(seq):
        return [x for i, x in enumerate(seq) if x != -100 and (i == 0 or x != seq[i-1])]
    pred_seq = compress(pred)
    label_seq = compress(label)
    sm = SequenceMatcher(None, pred_seq, label_seq)
    return 1.0 - sm.ratio()

def segment_f1(preds, labels, ignore_index=-100):
    preds = preds[labels != ignore_index]
    labels = labels[labels != ignore_index]
    return f1_score(labels.cpu(), preds.cpu(), average='macro')

def evaluate_stage(preds, labels):
    """Evaluate metrics for a specific stage"""
    acc = frame_accuracy(preds, labels)
    f1 = segment_f1(preds, labels)
    edit = compute_edit_distance(preds.tolist(), labels.tolist())
    return {
        'accuracy': acc,
        'f1_score': f1,
        'edit_distance': edit
    }

# ---------- Setup ---------- #
def setup_evaluation(feature_dir='dataset/features/resnet50', 
                    label_dir='dataset/labels',
                    model_path='MSTCN/models/mstcn_model.pth',
                    batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    dataset = SurgicalVideoDataset(feature_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = MS_TCN(num_stages=4, num_layers=10, num_f_maps=64, input_dim=2048, num_classes=35)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, dataloader, device

def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(15, 12))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    sns.heatmap(cm_normalized, 
                cmap="Blues", 
                xticklabels=range(35), 
                yticklabels=range(35), 
                annot=False, 
                fmt=".2f")
    
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Phase")
    plt.ylabel("True Phase")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # ---------- Setup ---------- #
    model, dataloader, device = setup_evaluation()
    
    # ---------- Inference ---------- #
    all_preds = []
    all_targets = []
    stage_preds = [[] for _ in range(4)]  # For each stage
    
    print("Running inference...")
    with torch.no_grad():
        for feats, labels in tqdm(dataloader):
            feats = feats.permute(0, 2, 1).to(device)  # (B, D, T)
            labels = labels.to(device)
            
            outputs = model(feats)  # (S, B, C, T)
            
            # Get predictions for each stage
            for s in range(outputs.size(0)):
                stage_output = outputs[s].permute(0, 2, 1)  # (B, T, C)
                stage_pred = stage_output.argmax(dim=-1)  # (B, T)
                
                for pred_seq, label_seq in zip(stage_pred, labels):
                    mask = label_seq != -100
                    stage_preds[s].extend(pred_seq[mask].tolist())
            
            # Use final stage predictions for overall evaluation
            final_preds = outputs[-1].permute(0, 2, 1).argmax(dim=-1)  # (B, T)
            
            for pred_seq, label_seq in zip(final_preds, labels):
                mask = label_seq != -100
                all_preds.extend(pred_seq[mask].tolist())
                all_targets.extend(label_seq[mask].tolist())
    
    # ---------- Evaluation ---------- #
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    
    # Evaluate each stage
    print("\nStage-wise Evaluation:")
    for s in range(4):
        stage_metrics = evaluate_stage(torch.tensor(stage_preds[s]), all_targets)
        print(f"\nStage {s+1}:")
        print(f"Frame Accuracy: {stage_metrics['accuracy']:.4f}")
        print(f"Segmental F1: {stage_metrics['f1_score']:.4f}")
        print(f"Edit Distance Score: {stage_metrics['edit_distance']:.4f}")
    
    # Final stage evaluation
    print("\nFinal Stage (Overall) Evaluation:")
    final_metrics = evaluate_stage(all_preds, all_targets)
    print(f"Frame Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Segmental F1: {final_metrics['f1_score']:.4f}")
    print(f"Edit Distance Score: {final_metrics['edit_distance']:.4f}")
    
    # ---------- Confusion Matrix ---------- #
    cm = confusion_matrix(all_targets, all_preds, labels=range(35))
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
