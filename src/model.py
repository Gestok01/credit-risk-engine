import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1]/"credit_risk_pipeline.pkl"

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
  
    X = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

    prob =float( model.predict_proba(X)[0][1])

    
    prediction = int(prob >= 0.5)

    return prediction, prob
