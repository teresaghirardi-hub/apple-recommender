"""
predict.py - Shared prediction logic used by both the API and Streamlit app.
"""

import logging
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# What each segment sees on the homepage
SEGMENT_CONTENT = {
    "Individual": {
        "headline": "Welcome! Explore our latest devices.",
        "products": ["iPhone 15", "AirPods Pro", "Apple Watch Series 9"],
        "offer": "Trade in your old device and save up to $200.",
        "avg_order_value": "$1,543",       
        "priority": "Standard",            
    },
    "Business": {
        "headline": "Equip your team with Apple Business essentials.",
        "products": ["MacBook Pro (M3)", "iPad Pro", "Apple Business Manager"],
        "offer": "Volume licensing available — contact our Business team.",
        "avg_order_value": "$1,569",        
        "priority": "High — volume licensing potential",  
    },
    "Education": {
        "headline": "Special pricing for students and educators.",
        "products": ["MacBook Air (M2)", "iPad (10th Gen)", "Apple Pencil"],
        "offer": "Education bundle: save up to $300 with student verification.",
        "avg_order_value": "$1,607",        
        "priority": "Medium — discount sensitive",  
    },
    "Government": {
        "headline": "Procurement solutions for government agencies.",
        "products": ["Mac mini (M2)", "iPhone 15 Pro", "AppleCare+ for Enterprise"],
        "offer": "GSA Schedule pricing available. Request a quote.",
        "avg_order_value": "$1,554",        
        "priority": "High — bulk procurement",  
    },
}


def load_pipeline(artifact_path="models/pipeline.pkl"):
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {artifact_path}. Run src/train.py first."
        )
    pipeline = joblib.load(path)
    logger.info(f"Pipeline loaded from {artifact_path}")
    return pipeline


def predict_segment(pipeline, input_data: dict) -> dict:
    feature_order = [
        "product_name", "category", "color",
        "customer_age_group", "region", "country", "city"
    ]
    df = pd.DataFrame([input_data])[feature_order]

    segment      = pipeline.predict(df)[0]
    proba        = pipeline.predict_proba(df)[0]
    classes      = pipeline.classes_
    proba_dict   = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    content      = SEGMENT_CONTENT.get(segment, {})

    return {
        "segment":       segment,
        "probabilities": proba_dict,
        "content":       content,
    }