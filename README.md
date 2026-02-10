# url-police-ai

Phishing URL detection using a multi-view LightGBM ensemble.

Six separate LightGBM models are trained on feature groups (url, domain, directory, file, params, network) and combined via soft voting. This approach follows the "Explainable Multi-View Ensemble" methodology from the paper "Explainable Multi-View Ensemble Model for Phishing Website Detection" (2025), where each view specializes on a feature group to prevent dominant features from overshadowing weaker but informative signals.

## Pipeline

```
Raw URL → Feature Extraction (97 URL + 14 Network) → 6 Views → LightGBM per View → Soft Voting → Phishing Probability
```

**Performance:**
- **Training set:** F1=0.9126, AUC=0.9863 (129K samples)
- **Cross-dataset:** F1=0.7187, AUC=0.7911 (11K external samples, no network features)

## Quick Start

```bash
docker build -t url-police-ai .
docker run --rm -p 8000:8000 url-police-ai
```

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com"}'
```

## Key Findings

### ⚠️ Network Features Are Critical
- **Without network features:** Cross-dataset F1=0.7187, high false positives on Korean URLs (75%)
- **With network features:** Training F1=0.9126, AUC=0.9863
- **Recommendation:** Collect DNS/WHOIS/SSL data for production use (handled by Spring server)

### 🌏 Geographic Bias
- Model trained primarily on Western URLs
- Korean domains (.kr, .do endpoints, Korean banking patterns) trigger false positives
- **Solution:** Use `inference_improved.py` with Korean whitelist + weighted voting
  - Reduces Korean false positives: 75% → 0% (on tested samples)

### 📊 Per-View Analysis
- **URL/File views:** Overfire on Korean patterns (0.95-0.99 for both legit and phishing)
- **Domain view:** Best discriminator (0.17-0.21 legit vs 0.77 phishing)
- **Network view:** Strong predictor (F1=0.8412, AUC=0.9429) when features available

See `FALSE_POSITIVE_ANALYSIS.md` for detailed analysis.

## Setup (local)

```bash
pip install -r requirements.txt
```

## Dataset

**Training:** [Dataset Phishing Domain Detection - Cybersecurity](https://www.kaggle.com/datasets/michellevp/dataset-phishing-domain-detection-cybersecurity) (Kaggle)

- 129,698 samples (52,152 phishing / 77,546 legitimate)
- 111 features: 97 URL-derivable + 14 network features (DNS/WHOIS/SSL)
- Label: `phishing` (0 = legitimate, 1 = phishing)

**Test:** [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) (Kaggle)

- 11,430 samples with raw URLs
- Label: `status` ("legitimate" / "phishing")

## Features

111 total features: **97 URL-derivable** + **14 network features**

### URL-Derivable Features (97)
- **Per component** (url, domain, directory, file, params): counts of 17 special characters (`. - _ / ? = @ & ! [space] ~ , + * # $ %`) + length
- **Additional:** `qty_tld_url`, `qty_vowels_domain`, `domain_in_ip`, `server_client_domain`, `tld_present_params`, `qty_params`, `email_in_url`

### Network Features (14)
Require external DNS/WHOIS/SSL lookup:
- `time_response`, `domain_spf`, `asn_ip`, `time_domain_activation`, `time_domain_expiration`
- `qty_ip_resolved`, `qty_nameservers`, `qty_mx_servers`, `ttl_hostname`
- `tls_ssl_certificate`, `qty_redirects`, `url_google_index`, `domain_google_index`, `url_shortened`

Missing URL components (no path, no query string) use -1 for all their features.

### Feature Views (6 views)

| View | Features | Count |
|---|---|---|
| url | 17 char counts + qty_tld_url + length_url + email_in_url | 20 |
| domain | 17 char counts + vowels + domain_length + domain_in_ip + server_client_domain | 21 |
| directory | 17 char counts + directory_length | 18 |
| file | 17 char counts + file_length | 18 |
| params | 17 char counts + params_length + tld_present_params + qty_params | 20 |
| **network** | All 14 network features (DNS/WHOIS/SSL) | **14** |

**Note:** NETWORK view has the best individual performance (F1=0.9408, AUC=0.9868)

## Training

```bash
python train_ensemble.py --data dataset_cybersecurity_michelle.csv --val-ratio 0.2
```

Options:

```
--val-ratio   Validation split ratio (default: 0.1)
--out-dir     Output directory (default: checkpoints/ensemble)
```

Outputs:
- `checkpoints/ensemble/{view}_model.txt` - Per-view LightGBM models (6 files)
- `checkpoints/ensemble/config.json` - View configuration

## SHAP Explainability

Analyze feature importance using SHAP values:

```bash
# Analyze all views
python explainability.py --data dataset_cybersecurity_michelle.csv --save-dir shap_analysis

# Explain a single URL
python explainability.py --url "https://suspicious-site.com/login.php"
```

Key findings from paper:
- **Most important feature:** `time_domain_activation` (3.14) from NETWORK view
- **URL view:** `qty_slash_url` (2.64), `length_url` (1.85)
- **File view:** `qty_dot_file` (2.53)

## Evaluation

Cross-dataset evaluation on a separate test set with raw URLs:

```bash
python evaluate.py --test dataset_phishing.csv
```

## API

Run with Docker (see Quick Start) or locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/classify` | Classify a single URL (with optional network features) |

```bash
# Classify URL only (network features default to -1)
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/login"}'

# Classify with network features
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/login",
    "network_features": {
      "time_response": 0.245,
      "domain_spf": 1,
      "asn_ip": 13335,
      "time_domain_activation": 1640,
      "time_domain_expiration": 551
    }
  }'
# {"url": "...", "phishing_probability": 0.38, "is_phishing": false}
```

### Python module

**Standard inference:**
```python
from inference import classify_url, classify, classify_batch
import numpy as np

# URL-only inference (network features = -1)
score = classify_url("https://example.com/login")  # 0.0 = legit, 1.0 = phishing

# With network features
network_data = {
    "time_response": 0.245,
    "domain_spf": 1,
    "asn_ip": 13335,
    # ... other network features
}
score = classify_url("https://example.com/login", network_features=network_data)

# Direct feature array (111 features: 97 URL + 14 network)
features = np.array([...], dtype=np.float32)
score = classify(features)

batch = np.array([[...], [...]], dtype=np.float32)
scores = classify_batch(batch)
```

**Improved inference** (for production with Korean URLs):
```python
from inference_improved import classify_url_improved

# Returns detailed results with per-view probabilities
result = classify_url_improved("https://shinhan.com/banking")
# {
#   'probability': 0.30,
#   'is_phishing': False,
#   'view_probabilities': {'url': 0.90, 'domain': 0.18, ...},
#   'whitelist_match': True
# }
```

Features:
- Weighted view voting (reduces false positives on Korean patterns)
- Korean domain whitelist (banks, government, e-commerce)
- Adjusted threshold (0.55 instead of 0.5)
- Per-view probability breakdown for debugging

## Project Structure

```
app.py                      FastAPI service (with optional network_features)
features.py                 URL → 97 URL-derivable + 14 network features + FEATURE_VIEWS
inference.py                6-view ensemble inference (standard)
inference_improved.py       Improved inference with Korean domain support
train_ensemble.py           Multi-view LightGBM ensemble training (6 views)
explainability.py           SHAP-based feature importance analysis
evaluate.py                 Cross-dataset evaluation
test_improvements.py        Comparison test: original vs improved inference
model.py                    MLP model definition (reference)
train.py                    MLP training script (reference)
Dockerfile                  Container image
requirements.txt            Python dependencies (includes shap, matplotlib)
IMPROVEMENTS.md             Architecture improvement guide (5→6 views)
FALSE_POSITIVE_ANALYSIS.md  False positive analysis and fixes
checkpoints/
  ensemble/
    {view}_model.txt        Per-view LightGBM models (6 files)
    config.json             View configuration
```

## Performance

### Training Results (dataset_cybersecurity_michelle.csv)

**Ensemble (6-view soft voting):**
| Metric | Value |
|--------|-------|
| **Precision** | 0.8674 |
| **Recall** | 0.9627 |
| **F1-score** | 0.9126 |
| **ROC AUC** | 0.9863 |

**Per-View Performance:**
| View | F1-score | AUC | Features |
|------|----------|-----|----------|
| **url** | 0.8787 | 0.9670 | 20 |
| **domain** | 0.6700 | 0.7888 | 21 |
| **directory** | 0.8617 | 0.9511 | 18 |
| **file** | 0.8361 | 0.9127 | 18 |
| **params** | 0.3530 | 0.6079 | 20 |
| **network** | 0.8412 | 0.9429 | 14 |

- Training set: 103,758 samples (20% validation split)
- Validation set: 25,940 samples

### Cross-Dataset Evaluation (dataset_phishing.csv)

External test set with 11,430 samples (5,715 phishing / 5,715 legitimate):

| Metric | Value | Note |
|--------|-------|------|
| **Precision** | 0.6694 | - |
| **Recall** | 0.7757 | - |
| **F1-score** | 0.7187 | - |
| **ROC AUC** | 0.7911 | - |

**Confusion Matrix:**
```
TN=3526  FP=2189  (Legit correctly classified: 61.7%)
FN=1282  TP=4433  (Phishing correctly classified: 77.6%)
```

**Note:** Cross-dataset performance drops from F1=0.9126 → 0.7187 due to:
- Different URL patterns/distributions between datasets
- Network features unavailable (defaulted to -1)
- Geographic/temporal distribution shift

### False Positive Analysis (Korean URLs)

Tested on 12 legitimate Korean URLs (banks, government, e-commerce) with the base model:
- **False Positive Rate:** 75% (9/12 incorrectly flagged as phishing)
- **Root cause:** Training data lacks Korean domain patterns (.kr TLDs, .do endpoints, Korean URL shorteners)

**Improved inference** (`inference_improved.py`):
- Weighted view voting (reduce URL/File weights, increase Domain/Network)
- Korean domain whitelist (major banks, .kr TLDs, known legitimate domains)
- Adjusted threshold (0.55 instead of 0.5)
- **Result:** 0% false positive rate on tested Korean URLs (9/12 → 0/12)

See `FALSE_POSITIVE_ANALYSIS.md` for detailed analysis and improvement roadmap.

### Paper Baseline (for reference)

Paper "Explainable Multi-View Ensemble Model for Phishing Website Detection" (2025):
- Accuracy: 96.0%, F1: 0.9497, AUC: 0.9923
- Network view: F1=0.9408, AUC=0.9868 (best individual predictor)

## References

**Paper:** Hong, H., Park, J., & Jeon, S. (2025). "Explainable Multi-View Ensemble Model for Phishing Website Detection." *Journal of Internet of Things and Convergence*, 11(4), 143-149.
