# CLAUDE.md

## Project

Phishing URL detection service. Multi-view LightGBM ensemble trained on URL-structural features, served via FastAPI.

## Architecture

Pipeline: Raw URL → `features.py` (97 features) → 5 views → LightGBM per view → soft voting → probability.

Views: url(20), domain(21), directory(18), file(18), params(20) = 97 total features.

## Key Files

- `app.py` — FastAPI service (`POST /classify`, `GET /health`)
- `features.py` — URL → 97 features (`extract_features`, `FEATURE_NAMES`, `FEATURE_VIEWS`)
- `inference.py` — LightGBM ensemble inference (`classify_url`, `classify`, `classify_batch`)
- `train_ensemble.py` — Training script (5 per-view LightGBM models)
- `evaluate.py` — Cross-dataset evaluation
- `checkpoints/ensemble/` — Trained model files

## Commands

```bash
# Run API locally
uvicorn app:app --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t url-police-ai .
docker run --rm -p 8000:8000 url-police-ai

# Train models
python train_ensemble.py --data dataset_cybersecurity_michelle.csv

# Evaluate on test set
python evaluate.py --test dataset_phishing.csv
```

## Dependencies

Runtime: lightgbm, numpy (<2), fastapi, uvicorn (see `requirements.txt`).
Training: + pandas, scikit-learn.

## Notes

- Missing URL components (no path, no query) use -1 sentinel values; LightGBM handles these natively
- `model.py` and `train.py` are the old MLP approach, kept as reference
- Docker image requires `libgomp1` for LightGBM's OpenMP support
