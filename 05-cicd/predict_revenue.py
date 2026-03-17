"""
predict_revenue.py - Shared prediction logic for revenue model.
"""

import joblib
import pandas as pd


def load_revenue_pipeline(path="models/revenue_pipeline.pkl"):
    return joblib.load(path)


def predict_revenue(pipeline, input_data: dict) -> dict:
    """
    input_data keys:
        product_name, category, color, customer_age_group,
        region, country, city, sales_channel, payment_method,
        discount_pct, units_sold
    Returns:
        {"revenue_usd": float}
    """
    df = pd.DataFrame([input_data])
    predicted = pipeline.predict(df)[0]
    return {"revenue_usd": round(float(predicted), 2)}