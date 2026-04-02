import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

from src.model import build_model
from src.dataset import RedditMentalHealthDataset, ID2LABEL, NUM_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model():
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load('outputs/models/best_model.pt',
                                     map_location=DEVICE))
    model.eval()
    return model


def get_predictions(model, loader):
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion matrix (% of actual class)', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved → outputs/confusion_matrix.png")


def error_analysis(test_df, labels, preds, n=5):
    """Print n misclassified examples per class — shows where model struggles."""
    print("\n── Error Analysis (sample misclassifications) ──────────────────")
    test_df = test_df.copy().reset_index(drop=True)
    test_df['true_label'] = [ID2LABEL[i] for i in labels]
    test_df['pred_label'] = [ID2LABEL[i] for i in preds]
    wrong = test_df[test_df['true_label'] != test_df['pred_label']]

    for true_class in sorted(wrong['true_label'].unique()):
        subset = wrong[wrong['true_label'] == true_class].head(2)
        for _, row in subset.iterrows():
            print(f"\n  True: {row['true_label']}  |  Predicted: {row['pred_label']}")
            print(f"  Text: {row['text'][:180]}...")


def main():
    os.makedirs('outputs', exist_ok=True)

    print("Loading model...")
    model = load_best_model()

    print("Loading test set...")
    test_ds = RedditMentalHealthDataset('data/processed/test_clean.csv',
                                        max_length=128)
    test_loader = DataLoader(test_ds, batch_size=64,
                             shuffle=False, num_workers=0)

    print("Running inference on test set...")
    labels, preds, probs = get_predictions(model, test_loader)

    # ── Metrics ──
    class_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='weighted')

    print(f"\n── Test Results ─────────────────────────────────────")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {f1:.4f}")
    print(f"\n── Per-class Report ─────────────────────────────────")
    print(classification_report(labels, preds, target_names=class_names))

    # ── Confusion matrix ──
    plot_confusion_matrix(labels, preds, class_names)

    # ── Error analysis ──
    test_df = pd.read_csv('data/processed/test_clean.csv')
    error_analysis(test_df, labels, preds)

    # ── Save results to CSV for README ──
    report = classification_report(labels, preds,
                                   target_names=class_names,
                                   output_dict=True)
    results_df = pd.DataFrame(report).transpose()
    results_df.to_csv('outputs/test_results.csv')
    print("\nSaved → outputs/test_results.csv")


if __name__ == "__main__":
    main()