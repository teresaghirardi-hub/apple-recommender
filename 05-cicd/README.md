# 05 – CI/CD Pipeline

This module contains the automated CI/CD pipeline for the Apple Segment Classifier.

---

## Overview

Every push to `main` triggers the following pipeline:
```
Push to main
     │
     ▼
Lint with flake8
     │
     ▼
Run unit tests (pytest)
     │
     ▼
Build Docker image
     │
     ▼
Deploy to Render
```

---

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Builds the container, trains the model, starts the API |
| `train.py` | Training script called during Docker build |
| `requirements.txt` | All Python dependencies |
| `test_api.py` | Tests the live Render endpoint |

---

## How to run locally
```bash
# Build the image
docker build -f 05-cicd/Dockerfile -t apple-recommender .

# Run the container
docker run -p 8000:8000 apple-recommender
```

---

## Live endpoint

https://apple-recommender.onrender.com