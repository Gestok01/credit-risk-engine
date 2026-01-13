import shap
import pandas as pd
import joblib

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

explainer = shap.LinearExplainer(
    model.named_steps["model"],
    model.named_steps["imputer"].transform(
        pd.DataFrame([[0]*len(FEATURE_ORDER)], columns=FEATURE_ORDER)
    )
)

def shap_explain(features: dict):
    X = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

    X_transformed = model.named_steps["imputer"].transform(X)

    shap_values = explainer.shap_values(X_transformed)[0]

    return dict(zip(FEATURE_ORDER, shap_values))