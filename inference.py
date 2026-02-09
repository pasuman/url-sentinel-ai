"""
Inference module for multi-view LightGBM ensemble.

Loads 6 per-view LightGBM models (url, domain, directory, file, params, network)
and combines predictions via soft voting.

Network features require external data collection (DNS, WHOIS). For URL-only
inference, network features default to -1 (missing value marker).

Usage:
    from inference import classify_url

    # URL-only inference (network features = -1)
    score = classify_url("https://example.com")
    print(f"Phishing probability: {score:.4f}")

    # With network features
    network_data = {...}  # 14 network features
    score = classify_url_with_network("https://example.com", network_data)
"""

import os

import lightgbm as lgb
import numpy as np

from features import FEATURE_VIEWS, NETWORK_FEATURE_NAMES, extract_features

_ENSEMBLE_DIR = "checkpoints/ensemble"

_models: dict[str, lgb.Booster] | None = None
_view_indices: dict[str, list[int]] | None = None


def load_ensemble(
    ensemble_dir: str = _ENSEMBLE_DIR,
) -> tuple[dict[str, lgb.Booster], dict[str, list[int]]]:
    """Load all per-view LightGBM models."""
    global _models, _view_indices
    if _models is None:
        _models = {}
        for view_name in FEATURE_VIEWS:
            model_path = os.path.join(ensemble_dir, f"{view_name}_model.txt")
            _models[view_name] = lgb.Booster(model_file=model_path)
        _view_indices = dict(FEATURE_VIEWS)
    return _models, _view_indices


def classify(
    features: np.ndarray,
    ensemble_dir: str = _ENSEMBLE_DIR,
) -> float:
    """
    Classify a feature vector and return phishing probability.

    Args:
        features: 1D array of 111 numeric features (97 URL-derivable + 14 network).
        ensemble_dir: Path to ensemble model directory.

    Returns:
        Risk score between 0.0 (legitimate) and 1.0 (phishing).
    """
    models, view_indices = load_ensemble(ensemble_dir)
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    prob_sum = 0.0
    for view_name, indices in view_indices.items():
        prob_sum += models[view_name].predict(x[:, indices])[0]
    return float(prob_sum / len(view_indices))


def classify_batch(
    features_batch: np.ndarray,
    ensemble_dir: str = _ENSEMBLE_DIR,
) -> list[float]:
    """Classify a batch of feature vectors."""
    models, view_indices = load_ensemble(ensemble_dir)
    x = np.asarray(features_batch, dtype=np.float32)
    prob_sum = np.zeros(len(x))
    for view_name, indices in view_indices.items():
        prob_sum += models[view_name].predict(x[:, indices])
    probs = prob_sum / len(view_indices)
    return [float(p) for p in probs]


def classify_url(
    url: str,
    ensemble_dir: str = _ENSEMBLE_DIR,
    network_features: np.ndarray | dict[str, float] | None = None,
) -> float:
    """
    Classify a raw URL and return phishing probability.

    Extracts 97 URL-derivable features and combines with network features
    (if provided). Network features default to -1 (missing).

    Args:
        url: Raw URL string (e.g., "https://example.com/path?q=1").
        ensemble_dir: Path to ensemble model directory.
        network_features: Optional network features (14 values) as array or dict.
            If None, all network features are set to -1.

    Returns:
        Risk score between 0.0 (legitimate) and 1.0 (phishing).
    """
    # Extract URL-derivable features (97)
    url_features = extract_features(url)

    # Handle network features
    if network_features is None:
        # Use -1 as missing value marker for all network features
        net_array = np.full(len(NETWORK_FEATURE_NAMES), -1.0, dtype=np.float32)
    elif isinstance(network_features, dict):
        # Convert dict to array in the correct order
        net_array = np.array(
            [network_features.get(name, -1.0) for name in NETWORK_FEATURE_NAMES],
            dtype=np.float32,
        )
    else:
        net_array = np.asarray(network_features, dtype=np.float32)

    # Combine: 97 URL-derivable + 14 network = 111 total
    combined_features = np.concatenate([url_features, net_array])

    return classify(combined_features, ensemble_dir)
