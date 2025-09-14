#!/usr/bin/env python3
"""
A PyTorch script to train a waste image classifier using transfer learning.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform

def stratified_index_split(targets: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
        
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def loop_one_epoch(model, loader, device, optimizer=None, scaler=None, scheduler=None, sched_per_batch=False):
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss, total_acc = 0.0, 0.0
    pbar = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
        if is_train:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if sched_per_batch and scheduler:
                scheduler.step()
                
        acc = (logits.argmax(1) == labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / len(loader), total_acc / len(loader)

def save_results(history, cm, report, class_names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save curves
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()
    
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(val_acc_history, label="val_acc")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend(), plt.tight_layout()
    plt.savefig(out_dir / "acc_curve.png", dpi=150)
    plt.close()
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    # Save classification report
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

def main(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_transform, eval_transform = make_transforms(args.img_size)

    full_train_dataset = ImageFolder(args.train_dir, transform=train_transform)
    test_dataset = ImageFolder(args.test_dir, transform=eval_transform)
    
    if set(full_train_dataset.classes) != set(test_dataset.classes):
        raise ValueError("Train and test folders have different classes!")

    targets = [y for _, y in full_train_dataset.samples]
    train_indices, val_indices = stratified_index_split(targets, args.val_frac, args.seed)
    
    val_dataset = ImageFolder(args.train_dir, transform=eval_transform)

    train_loader = DataLoader(Subset(full_train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    class_names = full_train_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Data split -> Train: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_dataset)}")

    model = build_model(args.arch, len(class_names)).to(device)
    
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if arch.startswith("resnet") and not name.startswith("fc."):
                param.requires_grad = False
            elif arch.startswith("efficientnet") and not name.startswith("classifier."):
                param.requires_grad = False
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    
    if args.onecycle:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
        sched_per_batch = True
    else:
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        sched_per_batch = False
        
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    best_val_acc = -1.0
    no_improve_epochs = 0
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = loop_one_epoch(model, train_loader, device, optimizer, scaler, scheduler, sched_per_batch)
        val_loss, val_acc = loop_one_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if not sched_per_batch:
            scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pth")
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs with no improvement.")
            break
            
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(args.out_dir / "best_model.pth"))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    history = {
        "train_loss": train_loss_history, "val_loss": val_loss_history,
        "train_acc": train_acc_history, "val_acc": val_acc_history
    }
    save_results(history, cm, report, class_names, args.out_dir)
    
    print("\n--- Test Set Results ---")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a waste classifier.")
    parser.add_argument("--train_dir", type=Path, required=True, help="Path to the training data directory.")
    parser.add_argument("--test_dir", type=Path, required=True, help="Path to the test data directory.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to save outputs.")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--step", type=int, default=10, help="StepLR step size.")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma factor.")
    parser.add_argument("--onecycle", action="store_true", help="Use OneCycleLR instead of StepLR.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction of training data for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone and train only the classifier head.")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
