import numpy as np
from src.model import model, FEATURE_ORDER


FEATURE_LABELS = {
    "num_late_payments": "Many late payments",
    "avg_delay": "High average delay",
    "credit_utilisation": "High credit utilisation",
    "payment_ratio": "Low payment ratio",
    "high_risk_flag": "High risk behavioural flag",
    "LIMIT_BAL": "Low credit limit",
    "AGE": "Young age"
}

def explain_prediction(features: dict, top_k: int = 3):
    """
    Explain prediction using logistic regression coefficients
    """

    classifier = model.named_steps["model"]

    coefs = classifier.coef_[0]

    values = np.array([features[f] for f in FEATURE_ORDER])
    contributions = coefs * values

    explanations = []
    for f, c in zip(FEATURE_ORDER, contributions):
        explanations.append({
            "feature": f,
            "reason": FEATURE_LABELS[f],
            "impact": float(c)
        })

    explanations = sorted(
        explanations,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return explanations[:top_k]