"""
Multi-view LightGBM ensemble training script.

Trains separate LightGBM models per URL component group (url, domain,
directory, file, params, network), then combines via soft voting.

6-view ensemble as per "Explainable Multi-View Ensemble Model for Phishing Website Detection":
- Views: url(20), domain(21), directory(18), file(18), params(20), network(14)
- Total: 111 features (97 URL-derivable + 14 network)

Usage:
    python train_ensemble.py --data dataset_cybersecurity_michelle.csv
"""

import argparse
import json
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from features import FEATURE_NAMES, FEATURE_VIEWS, NETWORK_FEATURE_NAMES

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train multi-view LightGBM ensemble")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", default="checkpoints/ensemble")
    args = parser.parse_args()

    set_seed()

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Shape: {df.shape}")

    # Build combined feature array: 97 URL-derivable + 14 network = 111 total
    # URL-derivable features are columns 1-97 (including server_client_domain at 40)
    # Network features are columns 98-111
    all_feature_names = FEATURE_NAMES + NETWORK_FEATURE_NAMES

    # Check if all required columns exist
    missing = [col for col in all_feature_names if col not in df.columns]
    if missing:
        print(f"WARNING: Missing columns in dataset: {missing}")
        print(f"Will proceed with available features only")
        all_feature_names = [col for col in all_feature_names if col in df.columns]

    y = df["phishing"].values.astype(np.float32)
    X = df[all_feature_names].values.astype(np.float32)

    print(f"Features: {X.shape[1]} (97 URL-derivable + {len(NETWORK_FEATURE_NAMES)} network)")
    print(f"Phishing: {int(y.sum())} | Legit: {int(len(y) - y.sum())}")

    # Shuffle and split (same seed/ratio as MLP training)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - args.val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Train per-view models
    val_probs_all = np.zeros(len(X_val))
    view_metrics = {}

    for view_name, view_indices in FEATURE_VIEWS.items():
        print(f"\n--- Training [{view_name}] view ({len(view_indices)} features) ---")

        # Get feature names for this view
        view_feature_names = [all_feature_names[i] for i in view_indices]

        X_train_view = X_train[:, view_indices]
        X_val_view = X_val[:, view_indices]

        train_data = lgb.Dataset(
            X_train_view, label=y_train,
            feature_name=view_feature_names,
        )
        val_data = lgb.Dataset(
            X_val_view, label=y_val,
            feature_name=view_feature_names,
            reference=train_data,
        )

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "is_unbalance": True,
            "seed": SEED,
            "verbose": -1,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Per-view validation
        val_probs = model.predict(X_val_view)
        val_preds = (val_probs >= 0.5).astype(int)
        metrics = {
            "precision": precision_score(y_val, val_preds, zero_division=0),
            "recall": recall_score(y_val, val_preds, zero_division=0),
            "f1": f1_score(y_val, val_preds, zero_division=0),
            "roc_auc": roc_auc_score(y_val, val_probs),
            "n_features": len(view_indices),
            "best_iteration": model.best_iteration,
        }
        view_metrics[view_name] = metrics
        print(
            f"  P: {metrics['precision']:.4f}  "
            f"R: {metrics['recall']:.4f}  "
            f"F1: {metrics['f1']:.4f}  "
            f"AUC: {metrics['roc_auc']:.4f}  "
            f"(iter {metrics['best_iteration']})"
        )

        val_probs_all += val_probs

        # Save model
        model_path = os.path.join(args.out_dir, f"{view_name}_model.txt")
        model.save_model(model_path)
        print(f"  Saved: {model_path}")

    # Ensemble (soft voting = average probabilities)
    n_views = len(FEATURE_VIEWS)
    ensemble_probs = val_probs_all / n_views
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    ensemble_metrics = {
        "precision": precision_score(y_val, ensemble_preds, zero_division=0),
        "recall": recall_score(y_val, ensemble_preds, zero_division=0),
        "f1": f1_score(y_val, ensemble_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val, ensemble_probs),
    }

    print("\n=== Ensemble (soft voting) ===")
    print(
        f"  P: {ensemble_metrics['precision']:.4f}  "
        f"R: {ensemble_metrics['recall']:.4f}  "
        f"F1: {ensemble_metrics['f1']:.4f}  "
        f"AUC: {ensemble_metrics['roc_auc']:.4f}"
    )

    # Summary
    print("\n=== Per-View Summary ===")
    for name, m in view_metrics.items():
        print(f"  {name:>10s}: F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  ({m['n_features']} features)")

    # Save view config for inference
    config = {
        "views": {name: indices for name, indices in FEATURE_VIEWS.items()},
        "n_features_total": len(all_feature_names),
        "n_url_derivable": len(FEATURE_NAMES),
        "n_network": len(NETWORK_FEATURE_NAMES),
        "url_derivable_features": FEATURE_NAMES,
        "network_features": NETWORK_FEATURE_NAMES,
    }
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")


if __name__ == "__main__":
    main()
