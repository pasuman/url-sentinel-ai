"""
Training script for PhishingMLP.

Expects a CSV where all columns are numeric features except the last column 'phishing' (label).

Usage:
    python train.py --data dataset_cybersecurity_michelle.csv
    python train.py --data dataset.csv --epochs 10 --batch-size 512 --lr 1e-3
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model import PhishingMLP

SEED = 42

# External/network columns that require runtime data collection (DNS, WHOIS, etc.)
# server_client_domain is NOW URL-derivable (moved out of external)
EXTERNAL_COLUMNS = [
    "time_response",
    "domain_spf",
    "asn_ip",
    "time_domain_activation",
    "time_domain_expiration",
    "qty_ip_resolved",
    "qty_nameservers",
    "qty_mx_servers",
    "ttl_hostname",
    "tls_ssl_certificate",
    "qty_redirects",
    "url_google_index",
    "domain_google_index",
    "url_shortened",  # also part of network view now
]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_probs.extend(model(xb).numpy())
            all_labels.extend(yb.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser(description="Train PhishingMLP")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    set_seed()

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Shape: {df.shape}")

    drop_cols = [c for c in EXTERNAL_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped {len(drop_cols)} external columns: {drop_cols}")

    y = df["phishing"].values.astype(np.float32)
    X = df.drop(columns=["phishing"]).values.astype(np.float32)
    feature_names = [c for c in df.columns if c != "phishing"]
    input_dim = X.shape[1]

    print(f"Features: {input_dim} | Phishing: {int(y.sum())} | Legit: {int(len(y) - y.sum())}")

    # Shuffle and split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - args.val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Save scaler + feature names
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "scaler.npz"),
        mean=mean,
        std=std,
        feature_names=np.array(feature_names),
    )

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Class imbalance handling
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if pos_weight > 1.5 or pos_weight < 0.67:
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        print(f"Using weighted sampler (pos_weight={pos_weight:.2f})")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model = PhishingMLP(input_dim)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, val_loader)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Loss: {total_loss / n_batches:.4f} | "
            f"P: {metrics['precision']:.4f} "
            f"R: {metrics['recall']:.4f} "
            f"F1: {metrics['f1']:.4f} "
            f"AUC: {metrics['roc_auc']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

    # Final eval with best model
    model.load_state_dict(
        torch.load(os.path.join(args.out_dir, "best_model.pt"), weights_only=True)
    )
    final = evaluate(model, val_loader)
    print("\n--- Best Model ---")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    # ONNX export
    model.eval()
    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(
        model,
        dummy,
        os.path.join(args.out_dir, "phishing_mlp.onnx"),
        input_names=["features"],
        output_names=["probability"],
        dynamic_axes={"features": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nONNX exported to {args.out_dir}/phishing_mlp.onnx")
    print(f"Scaler saved to {args.out_dir}/scaler.npz")


if __name__ == "__main__":
    main()
