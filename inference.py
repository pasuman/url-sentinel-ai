"""
Inference module for multi-view LightGBM ensemble.

Loads 5 per-view LightGBM models and combines predictions via soft voting.

Usage:
    from inference import classify_url

    score = classify_url("https://example.com")
    print(f"Phishing probability: {score:.4f}")
"""

import os

import lightgbm as lgb
import numpy as np

from features import FEATURE_VIEWS, extract_features

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
        features: 1D array of 97 numeric features.
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
) -> float:
    """
    Classify a raw URL and return phishing probability.

    Extracts 97 URL-derivable features, splits into views, and runs
    per-view LightGBM models with soft voting.

    Args:
        url: Raw URL string (e.g., "https://example.com/path?q=1").
        ensemble_dir: Path to ensemble model directory.

    Returns:
        Risk score between 0.0 (legitimate) and 1.0 (phishing).
    """
    features = extract_features(url)
    return classify(features, ensemble_dir)
