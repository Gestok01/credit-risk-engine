from fastapi import FastAPI
from src.schemas import CreditRequest, CreditResponse
from src.model import predict_risk

app = FastAPI(title="Credit Risk API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=CreditResponse)
def predict(req: CreditRequest):
    prediction, prob = predict_risk(req.dict())

    # Risk band logic
    if prob < 0.2:
        risk_band = "LOW"
    elif prob < 0.5:
        risk_band = "MEDIUM"
    else:
        risk_band = "HIGH"

    decision = "REJECT" if prediction == 1 else "APPROVE"

    return {
        "default_prediction": prediction,
        "probability_of_default": prob,
        "risk_band": risk_band,
        "decision": decision
    }
