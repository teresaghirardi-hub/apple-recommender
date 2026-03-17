# 🍎 Apple Segment Classifier

Real-time customer segment classifier for Apple homepage personalization.
Classifies visitors into **Individual, Business, Education, or Government**
and serves personalized product recommendations.

**Team 2:** Marcos Ortiz, Nuria Díaz Jiménez, Siddharth Murali, Teresa Ghirardi, Dan Tigu
**Course:** Machine Learning Operations — IE University

---

## Project Structure
```
apple-recommender/
├── data/               # Raw CSV (gitignored)
├── models/             # Saved pipeline.pkl (gitignored)
├── notebooks/          # EDA and KPI analysis
├── reports/            # Generated drift reports (gitignored)
├── src/
│   ├── train.py        # Training + MLflow tracking
│   ├── predict.py      # Shared prediction logic
│   └── monitor.py      # Evidently drift detection
├── tests/
│   └── test_predict.py
├── app.py              # Streamlit UI
├── api.py              # FastAPI endpoint
├── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model
```bash
python src/train.py
```

### 2. View MLflow experiments
```bash
mlflow ui
```
Open http://localhost:5000

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Run the API
```bash
uvicorn api:app --reload
```
- `GET  /health`  — health check
- `POST /predict` — segment prediction

### 5. Run tests
```bash
pytest tests/
``
### 6. Train the revenue model
```
python src/train_revenue.py
```
This trains a Random Forest Regressor to predict `revenue_usd` and saves `models/revenue_pipeline.pkl`.
