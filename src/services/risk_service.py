from src.config import RISK_THRESHOLDS,DECISION_RULES
def assess_risk(prob:float)->  str:
    if prob<RISK_THRESHOLDS["LOW"]:
        return "LOW"
    elif prob<RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "HIGH"
def make_decision(risk_band:str)->str:
    return DECISION_RULES[risk_band]