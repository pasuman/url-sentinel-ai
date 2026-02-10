"""
Improved inference with Korean domain awareness and view weighting.

Fixes for false positives:
1. Weighted voting (reduce URL/File, increase Domain/Network)
2. Korean TLD whitelist
3. Known legitimate domain patterns
4. Adjusted threshold
"""

import os
import re

import lightgbm as lgb
import numpy as np

from features import FEATURE_VIEWS, NETWORK_FEATURE_NAMES, extract_features

_ENSEMBLE_DIR = "checkpoints/ensemble"

# Korean legitimate TLDs and patterns
KOREAN_TLDS = {'.kr', '.co.kr', '.go.kr', '.or.kr', '.ac.kr'}
KNOWN_LEGITIMATE_DOMAINS = {
    'shinhan.com', 'kbcard.com', 'wooribank.com', 'naver.com',
    'kakao.com', 'toss.im', 'cjlogistics.com', 'emart.com',
    'seoul.go.kr', 'nwon.wooribank.com', 'meritzfire.com',
    'vo.la',  # Korean URL shortener
}

# View weights based on Korean context
# Reduce URL/File (overfire on Korean patterns), increase Domain/Network
VIEW_WEIGHTS = {
    'url': 0.6,       # Original high false positives
    'domain': 1.5,    # Best discriminator for Korean domains
    'directory': 0.8,
    'file': 0.5,      # Overreacts to .do endpoints
    'params': 1.0,
    'network': 2.0,   # Strongest predictor when available
}

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


def is_korean_legitimate_domain(url: str) -> bool:
    """Check if URL is from a known Korean legitimate domain or TLD."""
    url_lower = url.lower()

    # Check Korean TLDs
    for tld in KOREAN_TLDS:
        if tld in url_lower:
            return True

    # Check known legitimate domains
    for domain in KNOWN_LEGITIMATE_DOMAINS:
        if domain in url_lower:
            return True

    return False


def classify_weighted(
    features: np.ndarray,
    ensemble_dir: str = _ENSEMBLE_DIR,
    view_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Classify with weighted view voting.

    Returns:
        dict with 'probability', 'is_phishing', and per-view probabilities
    """
    if view_weights is None:
        view_weights = VIEW_WEIGHTS

    models, view_indices = load_ensemble(ensemble_dir)
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)

    view_probs = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for view_name, indices in view_indices.items():
        prob = float(models[view_name].predict(x[:, indices])[0])
        view_probs[view_name] = prob

        weight = view_weights.get(view_name, 1.0)
        weighted_sum += prob * weight
        total_weight += weight

    ensemble_prob = weighted_sum / total_weight

    return {
        'probability': ensemble_prob,
        'is_phishing': ensemble_prob > 0.55,  # Adjusted threshold (balance FP/FN)
        'view_probabilities': view_probs,
    }


def classify_url_improved(
    url: str,
    ensemble_dir: str = _ENSEMBLE_DIR,
    network_features: np.ndarray | dict[str, float] | None = None,
    use_whitelist: bool = True,
) -> dict[str, float]:
    """
    Improved URL classification with Korean domain awareness.

    Args:
        url: Raw URL string
        ensemble_dir: Path to ensemble models
        network_features: Optional network features (14 values)
        use_whitelist: Apply Korean domain whitelist

    Returns:
        dict with 'probability', 'is_phishing', 'view_probabilities', 'whitelist_match'
    """
    # Check whitelist first
    whitelist_match = False
    if use_whitelist and is_korean_legitimate_domain(url):
        whitelist_match = True

    # Extract features
    url_features = extract_features(url)

    # Handle network features
    if network_features is None:
        net_array = np.full(len(NETWORK_FEATURE_NAMES), -1.0, dtype=np.float32)
    elif isinstance(network_features, dict):
        net_array = np.array(
            [network_features.get(name, -1.0) for name in NETWORK_FEATURE_NAMES],
            dtype=np.float32,
        )
    else:
        net_array = np.asarray(network_features, dtype=np.float32)

    combined_features = np.concatenate([url_features, net_array])

    # Get weighted prediction
    result = classify_weighted(combined_features, ensemble_dir)
    result['whitelist_match'] = whitelist_match

    # Override if whitelisted
    if whitelist_match:
        result['probability'] = min(result['probability'], 0.3)  # Cap at 30%
        result['is_phishing'] = False

    return result


def classify_batch_improved(
    urls: list[str],
    ensemble_dir: str = _ENSEMBLE_DIR,
    use_whitelist: bool = True,
) -> list[dict[str, float]]:
    """Classify multiple URLs with improved logic."""
    return [classify_url_improved(url, ensemble_dir, use_whitelist=use_whitelist)
            for url in urls]
