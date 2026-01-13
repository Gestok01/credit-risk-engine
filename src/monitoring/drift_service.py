import pandas as pd
import joblib
from src.monitoring.psi import population_stability_index

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

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    drift_report = {}

    for feature in FEATURE_ORDER:
        psi_value = population_stability_index(
            reference_df[feature],
            current_df[feature]
        )

        drift_report[feature] = {
            "psi": round(psi_value, 4),
            "status": (
                "NO_DRIFT" if psi_value < 0.1 else
                "MODERATE_DRIFT" if psi_value < 0.25 else
                "SEVERE_DRIFT"
            )
        }

    return drift_report