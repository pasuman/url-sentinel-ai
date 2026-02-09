# CLAUDE.md

## Project

Phishing URL detection service. Multi-view LightGBM ensemble trained on URL-structural and network features, served via FastAPI.

Based on "Explainable Multi-View Ensemble Model for Phishing Website Detection" (2025).

## Architecture

Pipeline: Raw URL → `features.py` (97 URL features) + network data (14 features) → 6 views → LightGBM per view → soft voting → probability.

**6 Views:** url(20), domain(21), directory(18), file(18), params(20), network(14) = 111 total features.
- **URL-derivable**: 97 features (no external data needed)
- **Network**: 14 features (DNS, WHOIS, SSL - requires external lookup)

## Key Files

- `app.py` — FastAPI service (`POST /classify`, `GET /health`)
- `features.py` — URL → 97 URL-derivable features + 14 network features (`extract_features`, `FEATURE_NAMES`, `NETWORK_FEATURE_NAMES`, `FEATURE_VIEWS`)
- `inference.py` — LightGBM ensemble inference with optional network features (`classify_url`, `classify`, `classify_batch`)
- `train_ensemble.py` — Training script (6 per-view LightGBM models)
- `explainability.py` — SHAP-based feature importance analysis and visualization
- `evaluate.py` — Cross-dataset evaluation
- `checkpoints/ensemble/` — Trained model files (6 models: url, domain, directory, file, params, network)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn app:app --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t url-police-ai .
docker run --rm -p 8000:8000 url-police-ai

# Train 6-view ensemble
python train_ensemble.py --data dataset_cybersecurity_michelle.csv --val-ratio 0.2

# SHAP explainability analysis
python explainability.py --data dataset_cybersecurity_michelle.csv --save-dir shap_analysis

# Explain single URL
python explainability.py --url "https://suspicious-site.com/login.php"

# Evaluate on test set
python evaluate.py --test dataset_phishing.csv
```

## Dependencies

Runtime: lightgbm, numpy (<2), fastapi, uvicorn, shap, matplotlib (see `requirements.txt`).
Training: + pandas, scikit-learn.

## Notes

- Missing URL components (no path, no query) use -1 sentinel values; LightGBM handles these natively
- Network features default to -1 when unavailable (URL-only inference mode)
- NETWORK view is the strongest individual predictor (paper: F1=0.9408, AUC=0.9868)
- Expected ensemble performance (paper): Accuracy=96.0%, F1=0.949, AUC=0.9923
- `model.py` and `train.py` are the old MLP approach, kept as reference
- Docker image requires `libgomp1` for LightGBM's OpenMP support

## Feature Groups

### URL-Derivable Features (97 total)
- **URL** (20): 17 special chars + qty_tld_url + length_url + email_in_url
- **Domain** (21): 17 special chars + qty_vowels + domain_length + domain_in_ip + server_client_domain
- **Directory** (18): 17 special chars + directory_length
- **File** (18): 17 special chars + file_length
- **Params** (20): 17 special chars + params_length + tld_present_params + qty_params

### Network Features (14 total - require external lookup)
- time_response, domain_spf, asn_ip
- time_domain_activation, time_domain_expiration
- qty_ip_resolved, qty_nameservers, qty_mx_servers
- ttl_hostname, tls_ssl_certificate, qty_redirects
- url_google_index, domain_google_index, url_shortened
