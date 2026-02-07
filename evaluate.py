"""
Cross-dataset evaluation script.

Loads a test CSV with raw URLs and status labels, extracts features,
runs the multi-view LightGBM ensemble, and reports classification metrics.

Usage:
    python evaluate.py --test dataset_phishing.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import extract_features
from inference import classify_batch, load_ensemble


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set with raw URLs")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--url-col", default="url", help="Column name for URLs")
    parser.add_argument("--label-col", default="status", help="Column name for labels")
    parser.add_argument("--ensemble-dir", default="checkpoints/ensemble")
    args = parser.parse_args()

    print("Loading test data...")
    df = pd.read_csv(args.test)
    print(f"Shape: {df.shape}")

    # Map labels
    label_map = {"phishing": 1, "legitimate": 0}
    labels = df[args.label_col].map(label_map)
    if labels.isna().any():
        unknown = df[args.label_col][labels.isna()].unique()
        print(f"Warning: unknown labels {unknown}, dropping {labels.isna().sum()} rows")
        df = df[labels.notna()]
        labels = labels.dropna()
    y = labels.values.astype(np.float32)
    print(f"Phishing: {int(y.sum())} | Legit: {int(len(y) - y.sum())}")

    # Extract features
    print("Extracting features...")
    urls = df[args.url_col].values
    features = np.stack([extract_features(u) for u in urls])
    print(f"Features shape: {features.shape}")

    # Load ensemble and run inference
    print("Running inference...")
    load_ensemble(args.ensemble_dir)
    probs = np.array(classify_batch(features, args.ensemble_dir))

    # Metrics
    preds = (probs >= 0.5).astype(int)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)

    print("\n--- Cross-Dataset Evaluation ---")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")


if __name__ == "__main__":
    main()
