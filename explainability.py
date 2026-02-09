"""
SHAP-based explainability for multi-view LightGBM ensemble.

Provides feature importance analysis and visualization using SHAP values,
as described in "Explainable Multi-View Ensemble Model for Phishing Website Detection".

Usage:
    from explainability import analyze_view_shap, analyze_ensemble_shap

    # Analyze a single view
    analyze_view_shap(
        model_path="checkpoints/ensemble/network_model.txt",
        X_val=X_val_network,
        feature_names=network_feature_names,
        save_path="shap_network.png"
    )

    # Analyze all views
    analyze_ensemble_shap(
        ensemble_dir="checkpoints/ensemble",
        X_val=X_val,  # Full 111-feature array
        save_dir="shap_analysis"
    )
"""

import os
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap

from features import FEATURE_NAMES, FEATURE_VIEWS, NETWORK_FEATURE_NAMES


def analyze_view_shap(
    model_path: str,
    X_val: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    save_path: str | None = None,
) -> dict[str, float]:
    """
    Analyze SHAP values for a single view model.

    Args:
        model_path: Path to LightGBM model file (.txt)
        X_val: Validation data for this view (N x num_features)
        feature_names: List of feature names for this view
        top_k: Number of top features to analyze
        save_path: Optional path to save visualization

    Returns:
        Dict mapping feature names to mean absolute SHAP values
    """
    # Load model
    model = lgb.Booster(model_file=model_path)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_val)

    # Get mean absolute SHAP values for ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create feature importance dict
    importance = {
        name: float(value) for name, value in zip(feature_names, mean_abs_shap)
    }

    # Sort by importance
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    )

    # Print top features
    view_name = Path(model_path).stem.replace("_model", "").upper()
    print(f"\n=== {view_name} View - Top {top_k} Features ===")
    for i, (name, value) in enumerate(sorted_importance.items(), 1):
        print(f"  {i:2d}. {name:30s}: {value:.4f}")

    # Visualization
    if save_path:
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_val,
            feature_names=feature_names,
            max_display=top_k,
            show=False,
        )
        plt.title(f"SHAP Feature Importance - {view_name} View")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP plot to {save_path}")

    return sorted_importance


def analyze_ensemble_shap(
    ensemble_dir: str,
    X_val: np.ndarray,
    save_dir: str | None = None,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    """
    Analyze SHAP values for all views in the ensemble.

    Args:
        ensemble_dir: Directory containing all view models
        X_val: Full validation data (N x 111 features)
        save_dir: Optional directory to save visualizations
        top_k: Number of top features per view

    Returns:
        Dict mapping view names to their feature importance dicts
    """
    all_feature_names = FEATURE_NAMES + NETWORK_FEATURE_NAMES

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_importance = {}

    for view_name, view_indices in FEATURE_VIEWS.items():
        model_path = os.path.join(ensemble_dir, f"{view_name}_model.txt")
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found: {model_path}")
            continue

        # Extract view-specific features
        X_val_view = X_val[:, view_indices]
        view_feature_names = [all_feature_names[i] for i in view_indices]

        # Analyze this view
        save_path = (
            os.path.join(save_dir, f"shap_{view_name}.png") if save_dir else None
        )

        importance = analyze_view_shap(
            model_path=model_path,
            X_val=X_val_view,
            feature_names=view_feature_names,
            top_k=top_k,
            save_path=save_path,
        )

        all_importance[view_name] = importance

    # Print summary across all views
    print("\n=== Cross-View Top Features ===")
    all_features = []
    for view_name, importance in all_importance.items():
        for feat_name, feat_value in importance.items():
            all_features.append((view_name, feat_name, feat_value))

    # Sort by SHAP value across all views
    all_features.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} features across all views:")
    for i, (view, name, value) in enumerate(all_features[:top_k], 1):
        print(f"  {i:2d}. [{view:9s}] {name:30s}: {value:.4f}")

    return all_importance


def explain_single_prediction(
    url: str,
    ensemble_dir: str = "checkpoints/ensemble",
    network_features: dict[str, float] | None = None,
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """
    Explain a single URL prediction using SHAP values.

    Shows which features contributed most to the phishing prediction.

    Args:
        url: URL to analyze
        ensemble_dir: Path to ensemble models
        network_features: Optional network features dict
        top_k: Number of top contributing features to show per view

    Returns:
        Dict mapping view names to list of (feature_name, shap_value) tuples
    """
    from inference import classify_url
    from features import extract_features

    # Get prediction
    prob = classify_url(url, ensemble_dir, network_features)
    print(f"URL: {url}")
    print(f"Phishing Probability: {prob:.4f} ({'PHISHING' if prob >= 0.5 else 'LEGIT'})\n")

    # Extract features
    url_features = extract_features(url)
    if network_features:
        net_array = np.array(
            [network_features.get(name, -1.0) for name in NETWORK_FEATURE_NAMES],
            dtype=np.float32,
        )
    else:
        net_array = np.full(len(NETWORK_FEATURE_NAMES), -1.0, dtype=np.float32)

    combined_features = np.concatenate([url_features, net_array])
    all_feature_names = FEATURE_NAMES + NETWORK_FEATURE_NAMES

    explanations = {}

    for view_name, view_indices in FEATURE_VIEWS.items():
        model_path = os.path.join(ensemble_dir, f"{view_name}_model.txt")
        if not os.path.exists(model_path):
            continue

        # Load model
        model = lgb.Booster(model_file=model_path)

        # Extract view features
        X_view = combined_features[view_indices].reshape(1, -1)
        view_feature_names = [all_feature_names[i] for i in view_indices]

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_view)[0]

        # Get top contributing features
        feature_contributions = list(zip(view_feature_names, shap_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        explanations[view_name] = feature_contributions[:top_k]

        # Print
        print(f"=== {view_name.upper()} View ===")
        for feat_name, shap_val in feature_contributions[:top_k]:
            direction = "→ PHISHING" if shap_val > 0 else "→ LEGIT"
            print(f"  {feat_name:30s}: {shap_val:+.4f} {direction}")
        print()

    return explanations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHAP analysis for multi-view ensemble")
    parser.add_argument("--ensemble-dir", default="checkpoints/ensemble")
    parser.add_argument("--data", required=True, help="Validation data CSV")
    parser.add_argument("--save-dir", default="shap_analysis")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--url", help="Analyze a single URL")

    args = parser.parse_args()

    if args.url:
        # Single URL explanation
        explain_single_prediction(args.url, args.ensemble_dir, top_k=args.top_k)
    else:
        # Full ensemble analysis
        import pandas as pd

        df = pd.read_csv(args.data)
        y = df["phishing"].values
        all_feature_names = FEATURE_NAMES + NETWORK_FEATURE_NAMES

        # Check available features
        available = [col for col in all_feature_names if col in df.columns]
        X = df[available].values.astype(np.float32)

        # Use a sample for SHAP (expensive to compute)
        sample_size = min(1000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]

        print(f"Analyzing SHAP values on {sample_size} validation samples...")
        analyze_ensemble_shap(
            ensemble_dir=args.ensemble_dir,
            X_val=X_sample,
            save_dir=args.save_dir,
            top_k=args.top_k,
        )

        print(f"\nSHAP visualizations saved to {args.save_dir}/")
