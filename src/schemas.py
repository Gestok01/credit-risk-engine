from pydantic import BaseModel
from typing import List, Dict, Any


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

class ExplainResponse(BaseModel):
     default_prediction: int 
     probability_of_default: float
     risk_band: str
     decision: str
     top_reasons:List[str]
     model:Dict[str,str]

class ComplianceResponse(BaseModel):
    status: str
    audit_summary: str
    compliance_status: str = "UNKNOWN"
    metrics: Dict[str, Any]
    flagged_cases_count: int = 0
    report_file: str = ""
    report_markdown: str


