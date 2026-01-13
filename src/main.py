from fastapi import FastAPI
from src.schemas import CreditRequest, CreditResponse , ExplainResponse
from src.model import predict_risk
from src.services.risk_service import assess_risk, make_decision
from src.config import MODEL_METADATA
from src.explain.explaination_service import explain_prediction
from src.explain.shap_service import shap_explain
from src.monitoring.drift_service import detect_drift 
from src.audit.audit_logger import log_decision
from src.services.retraining_service import evaluate_retraining
from src.audit.audit_logger import log_retraining_decision
import pandas as pd

app = FastAPI(title="Credit Risk API")


@app.get("/")
def health():
    return {"status": "ok",
            "model":MODEL_METADATA}

@app.post("/predict", response_model=CreditResponse)
def predict(req: CreditRequest):
    prediction, prob = predict_risk(req.dict())

    risk_band_value=assess_risk(prob)
    decision=make_decision(risk_band_value)

    log_decision(
        input_features=req.dict(),
        prediction=prediction,
        probability=prob,
        risk_band=risk_band_value,
        decision=decision,
        model_version=MODEL_METADATA["version"]
    )

    return {
        "default_prediction": prediction,
        "probability_of_default": prob,
        "risk_band": risk_band_value,
        "decision": decision
    }
@app.post("/explain", response_model=ExplainResponse)
def explain(req: CreditRequest):

    prediction, prob = predict_risk(req.dict())

    risk_band = assess_risk(prob)
    decision = make_decision(risk_band)

    explanations = explain_prediction(req.dict())

    return {
        "default_prediction": prediction,
        "probability_of_default": prob,
        "risk_band": risk_band,
        "decision": decision,
        "top_reasons": [e["reason"] for e in explanations],
        "model": {
            "name": MODEL_METADATA["model_name"],
            "version": MODEL_METADATA["version"]
        }
    }
@app.post("/explain/shap")
def explain_shap(req: CreditRequest):

    prediction, prob = predict_risk(req.dict())
    risk_band = assess_risk(prob)
    decision = make_decision(risk_band)

    shap_vals = shap_explain(req.dict())

    top_drivers = sorted(
        shap_vals.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return {
        "prediction": prediction,
        "probability_of_default": prob,
        "risk_band": risk_band,
        "decision": decision,
        "shap_values": shap_vals,
        "top_drivers": [f.replace("_", " ").title() for f, _ in top_drivers]
    }
@app.post("/monitor/drift")
def monitor_drift(payload: dict):
    """
    payload = {
        "reference": [ {...}, {...} ],
        "current": [ {...}, {...} ]
    }
    """

    reference_df = pd.DataFrame(payload["reference"])
    current_df = pd.DataFrame(payload["current"])

    report = detect_drift(reference_df, current_df)

    return {
        "model_health": "OK" if all(
            v["status"] == "NO_DRIFT" for v in report.values()
        ) else "ATTENTION_REQUIRED",
        "feature_drift": report
    }
@app.post("/monitor/retraining-check")
def retraining_check(payload: dict):
    """
    Payload = output of /monitor/drift
    """

    decision = evaluate_retraining(payload)

    # AUDIT LOG
    log_retraining_decision(payload, decision)

    return decision