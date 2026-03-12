"""
app.py - Streamlit front-end for the Apple Segment Classifier.

Run:
    streamlit run app.py
"""

import streamlit as st
from src.predict import load_pipeline, predict_segment

st.set_page_config(
    page_title="Apple – Personalised Homepage",
    page_icon="🍎",
    layout="wide",
)

@st.cache_resource
def get_pipeline():
    return load_pipeline()

# ── Sidebar: questionnaire ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍎 Tell us about you")
    st.caption("Answer a few questions so we can personalise your experience.")

    product_name = st.selectbox("What are you interested in?", [
        "iPhone 15", "iPhone 15 Pro", "MacBook Air", "MacBook Pro",
        "iPad Pro", "iPad", "Apple Watch Series 9", "AirPods Pro",
        "Mac mini", "iMac",
    ])
    category = st.selectbox("Product category", [
        "iPhone", "Mac", "iPad", "Apple Watch", "AirPods", "Accessories"
    ])
    color = st.selectbox("Preferred color", [
        "Black", "White", "Silver", "Gold", "Space Gray", "Midnight", "Starlight"
    ])
    age_group = st.selectbox("Your age group", [
        "18–24", "25–34", "35–44", "45–54", "55+"
    ])
    region = st.selectbox("Your region", [
        "North America", "Europe", "Asia", "South America",
        "Oceania", "Africa", "Middle East"
    ])
    country = st.text_input("Country", value="United States")
    city    = st.text_input("City", value="New York")

    submitted = st.button("Show my homepage", use_container_width=True, type="primary")

# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("🍎 Your Personalised Apple Experience")

if not submitted:
    st.info("👈 Fill in the questionnaire on the left and click **Show my homepage**.")
    st.stop()

input_data = {
    "product_name":       product_name,
    "category":           category,
    "color":              color,
    "customer_age_group": age_group,
    "region":             region,
    "country":            country,
    "city":               city,
}

try:
    pipeline = get_pipeline()
    result   = predict_segment(pipeline, input_data)
except FileNotFoundError:
    st.error("Model not found. Please run `python src/train.py` first.")
    st.stop()

segment = result["segment"]
content = result["content"]
proba   = result["probabilities"]

st.subheader(content["headline"])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recommended for you")
    for p in content["products"]:
        st.markdown(f"- **{p}**")
    st.info(content["offer"])

with col2:
    st.subheader("Your segment")
    st.metric("Predicted", segment)
    st.caption("Confidence scores")
    for seg, prob in sorted(proba.items(), key=lambda x: -x[1]):
        st.progress(prob, text=f"{seg}: {prob:.0%}")