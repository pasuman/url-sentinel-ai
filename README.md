# url-police-ai

Phishing URL detection using a multi-view LightGBM ensemble.

Five separate LightGBM models are trained on URL component groups (url, domain, directory, file, params) and combined via soft voting. This approach follows the "Explainable Multi-View Ensemble" methodology, where each view specializes on a URL component to prevent dominant features from overshadowing weaker but informative signals.

## Pipeline

```
Raw URL → Feature Extraction (97 features) → 5 Views → LightGBM per View → Soft Voting → Phishing Probability
```

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
- 97 URL-derivable features used (14 external DNS/WHOIS features dropped)
- Label: `phishing` (0 = legitimate, 1 = phishing)

**Test:** [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) (Kaggle)

- 11,430 samples with raw URLs
- Label: `status` ("legitimate" / "phishing")

## Features

97 numeric features extracted from URL structure:

- **Per component** (url, domain, directory, file, params): counts of 17 special characters (`. - _ / ? = @ & ! [space] ~ , + * # $ %`) + length
- **Additional:** `qty_tld_url`, `qty_vowels_domain`, `domain_in_ip`, `tld_present_params`, `qty_params`, `email_in_url`, `url_shortened`

Missing URL components (no path, no query string) use -1 for all their features.

### Feature Views

| View | Features | Count |
|---|---|---|
| url | 17 char counts + qty_tld_url + length_url + email_in_url | 20 |
| domain | 17 char counts + vowels + domain_length + domain_in_ip + url_shortened | 21 |
| directory | 17 char counts + directory_length | 18 |
| file | 17 char counts + file_length | 18 |
| params | 17 char counts + params_length + tld_present_params + qty_params | 20 |

## Training

```bash
python train_ensemble.py --data dataset_cybersecurity_michelle.csv
```

Options:

```
--val-ratio   Validation split ratio (default: 0.1)
--out-dir     Output directory (default: checkpoints/ensemble)
```

Outputs:
- `checkpoints/ensemble/{view}_model.txt` - Per-view LightGBM models (5 files)
- `checkpoints/ensemble/config.json` - View configuration

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
| `POST` | `/classify` | Classify a single URL |

```bash
# Classify a URL
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/login"}'
# {"url": "...", "phishing_probability": 0.38, "is_phishing": false}
```

### Python module

```python
from inference import classify_url, classify, classify_batch
import numpy as np

score = classify_url("https://example.com/login")  # 0.0 = legit, 1.0 = phishing

features = np.array([...], dtype=np.float32)  # 97 features
score = classify(features)

batch = np.array([[...], [...]], dtype=np.float32)
scores = classify_batch(batch)
```

## Project Structure

```
app.py               FastAPI service
features.py          URL → 97 numeric features + FEATURE_VIEWS
inference.py         Ensemble inference (classify, classify_batch, classify_url)
train_ensemble.py    Multi-view LightGBM ensemble training
evaluate.py          Cross-dataset evaluation
model.py             MLP model definition (reference)
train.py             MLP training script (reference)
Dockerfile           Container image
requirements.txt     Python dependencies
checkpoints/
  ensemble/
    {view}_model.txt   Per-view LightGBM models (5 files)
    config.json        View configuration
```
