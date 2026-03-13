"""
test_api.py - Tests the live API endpoint.

Usage:
    python 05-cicd/test_api.py
"""

import requests

BASE_URL = "https://apple-recommender.onrender.com"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == True
    print(f"✅ Health check passed: {data}")

def test_predict():
    payload = {
        "product_name": "MacBook Pro",
        "category": "Mac",
        "color": "Silver",
        "customer_age_group": "25–34",
        "region": "North America",
        "country": "United States",
        "city": "New York"
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "segment" in data
    assert "probabilities" in data
    assert data["segment"] in ["Individual", "Business", "Education", "Government"]
    print(f"✅ Predict passed: segment={data['segment']}")
    print(f"   Probabilities: {data['probabilities']}")

if __name__ == "__main__":
    print("Testing live API...\n")
    test_health()
    test_predict()
    print("\n✅ All tests passed!")