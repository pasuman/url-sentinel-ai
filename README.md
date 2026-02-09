# url-police-ai

Phishing URL detection using a multi-view LightGBM ensemble.

Six separate LightGBM models are trained on feature groups (url, domain, directory, file, params, network) and combined via soft voting. This approach follows the "Explainable Multi-View Ensemble" methodology from the paper "Explainable Multi-View Ensemble Model for Phishing Website Detection" (2025), where each view specializes on a feature group to prevent dominant features from overshadowing weaker but informative signals.

## Pipeline

```
Raw URL → Feature Extraction (97 URL + 14 Network) → 6 Views → LightGBM per View → Soft Voting → Phishing Probability
```

**Expected Performance:** Accuracy 96.0%, F1-score 0.9497, AUC 0.9923

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

## Project Structure

```
app.py               FastAPI service (with optional network_features)
features.py          URL → 97 URL-derivable + 14 network features + FEATURE_VIEWS
inference.py         6-view ensemble inference (classify, classify_batch, classify_url)
train_ensemble.py    Multi-view LightGBM ensemble training (6 views)
explainability.py    SHAP-based feature importance analysis
evaluate.py          Cross-dataset evaluation
model.py             MLP model definition (reference)
train.py             MLP training script (reference)
Dockerfile           Container image
requirements.txt     Python dependencies (includes shap, matplotlib)
IMPROVEMENTS.md      Detailed improvement guide
checkpoints/
  ensemble/
    {view}_model.txt   Per-view LightGBM models (6 files)
    config.json        View configuration
```

## Performance

Based on paper results with 6-view ensemble:

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.0% |
| **F1-score** | 0.9497 |
| **Precision** | 0.9609 |
| **Recall** | 0.9387 |
| **ROC AUC** | 0.9923 |

Individual view performance:
- **NETWORK** (best): F1=0.9408, AUC=0.9868
- **URL**: F1=0.8808, AUC=0.9690
- **DIRECTORY**: F1=0.8582, AUC=0.9517
- **FILE**: F1=0.8362, AUC=0.9132
- **DOMAIN**: F1=0.6600, AUC=0.7927
- **PARAMS**: F1=0.3567, AUC=0.6092

## References

**Paper:** Hong, H., Park, J., & Jeon, S. (2025). "Explainable Multi-View Ensemble Model for Phishing Website Detection." *Journal of Internet of Things and Convergence*, 11(4), 143-149.
