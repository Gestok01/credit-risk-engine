import joblib
import pandas as pd

MODEL_PATH = "credit_risk_pipeline.pkl"

model = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "num_late_payments",
    "avg_delay",
    "credit_utilisation",
    "payment_ratio",
    "high_risk_flag",
    "LIMIT_BAL",
    "AGE"
]

def predict_risk(features : dict):
    """
    payload: dict coming from API request
    """

    # 1️⃣ Convert dict → DataFrame (CRITICAL)
    X = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # 2️⃣ Predict probability
    prob = model.predict_proba(X)[0][1]

    # 3️⃣ Binary decision
    prediction = int(prob >= 0.5)

    return prediction, prob
