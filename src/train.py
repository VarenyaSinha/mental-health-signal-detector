import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import wandb

from src.model import build_model
from src.dataset import RedditMentalHealthDataset, NUM_LABELS

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name":   "distilbert-base-uncased",
    "max_length":   128,
    "batch_size":   32,
    "epochs":       4,
    "lr":           2e-5,
    "warmup_ratio": 0.1,
    "seed":         42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Class weights (handles mild imbalance) ────────────────────────────────────
def compute_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df['label_id'].value_counts().sort_index().values
    total = counts.sum()
    weights = total / (NUM_LABELS * counts)   # inverse frequency
    return torch.tensor(weights, dtype=torch.float)


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1


# ── Main training loop ────────────────────────────────────────────────────────
def train():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # W&B init — tracks every run automatically
    wandb.init(project="mental-health-signal-detector", config=CONFIG)

    # Data
    train_ds = RedditMentalHealthDataset('data/processed/train_clean.csv', CONFIG['max_length'])
    val_ds   = RedditMentalHealthDataset('data/processed/val_clean.csv',   CONFIG['max_length'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = build_model().to(DEVICE)

    # Loss with class weights
    class_weights = compute_class_weights('data/processed/train_clean.csv').to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    total_steps   = len(train_loader) * CONFIG['epochs']
    warmup_steps  = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0

    for epoch in range(CONFIG['epochs']):
        # ── Train ──
        model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping — prevents exploding gradients with transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        train_f1   = f1_score(all_labels, all_preds, average='weighted')
        train_loss = total_loss / len(train_loader)

        # ── Validate ──
        val_loss, val_f1 = evaluate(model, val_loader, criterion)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}\n")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_f1":   train_f1,
            "val_loss":   val_loss,
            "val_f1":     val_f1,
        })

        # Save best model only
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), '/content/drive/MyDrive/mhsd_best_model.pt')
            print(f"  ✓ New best model saved (val F1: {val_f1:.4f})\n")
            

    wandb.finish()
    print(f"Training complete. Best Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()