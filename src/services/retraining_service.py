from src.config import RETRAINING_POLICY

def evaluate_retraining(drift_report: dict):
    """
    Decide whether retraining should be triggered
    """

    drifted_features = [
        f for f, v in drift_report["feature_drift"].items()
        if v["psi"] >= RETRAINING_POLICY["psi_threshold"]
    ]

    should_retrain = len(drifted_features) >= RETRAINING_POLICY["min_features_drifted"]

    return {
        "should_retrain": should_retrain,
        "drifted_features": drifted_features,
        "policy": RETRAINING_POLICY
    }