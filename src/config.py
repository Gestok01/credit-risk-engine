RISK_THRESHOLDS={
    "LOW":0.2,
    "MEDIUM":0.5
}
DECISION_RULES={
    "LOW":"APPROVE",
    "MEDIUM":"REVIEW",
    "HIGH":"REJECT"
}
MODEL_METADATA={
    "model_name": "Credit Risk Logistic Regression",
    "version": "v1.0.0",
    "framework":"scikit-learn",
}
RETRAINING_POLICY = {
    "psi_threshold": 0.25,      # strong drift
    "min_features_drifted": 2,  # how many features must drift
    "action": "RETRAIN"
}