from pydantic import BaseModel

class CreditRequest(BaseModel):
    num_late_payments: int
    avg_delay: float
    credit_utilisation: float
    payment_ratio: float
    high_risk_flag: int
    LIMIT_BAL: float
    AGE: int

class CreditResponse(BaseModel):
    default_prediction: int 
    probability_of_default: float
    risk_band: str
    decision: str
