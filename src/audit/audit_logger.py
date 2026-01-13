import json
import uuid
from datetime import datetime
from pathlib import Path

AUDIT_LOG_PATH = Path("logs/audit_logs.jsonl")
AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)

def log_decision(
    input_features: dict,
    prediction: int,
    probability: float,
    risk_band: str,
    decision: str,
    model_version: str
):
    record = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "input_features": input_features,
        "prediction": prediction,
        "probability": probability,
        "risk_band": risk_band,
        "decision": decision
    }

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_retraining_decision(drift_report, retraining_decision):
    record = {
        "event": "RETRAINING_EVALUATION",
        "timestamp": datetime.utcnow().isoformat(),
        "drift_report": drift_report,
        "decision": retraining_decision
    }

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")        