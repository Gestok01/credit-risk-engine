import os
import json
import uuid
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Adjust python path to be able to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 printing in Windows console to avoid encoding crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.compliance.config import AUDIT_LOG_PATH, COMPLIANCE_REPORT_DIR
from src.compliance.rag_service import vector_db
from src.compliance.auditor_agent import run_compliance_audit

def setup_mock_audit_logs():
    """
    Cleans and populates logs/audit_logs.jsonl with a set of synthetic credit decisions.
    We carefully structure this to simulate:
    1. Overall high rejection rates for seniors (Disparate Impact / Bias).
    2. Outlier cases where highly creditworthy seniors are rejected (Suspicious cases).
    """
    print("[Test] Setting up mock logs in", AUDIT_LOG_PATH)
    
    # Ensure directory exists
    Path(AUDIT_LOG_PATH).parent.mkdir(exist_ok=True)
    
    # Clear existing logs
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
        
    mock_records = []
    
    # --- REFERENCE GROUP: Younger Cohort (<60) ---
    # High approval rate (7 approved out of 10)
    young_cases = [
        {"age": 25, "util": 0.20, "late": 0, "decision": "APPROVE", "prob": 0.05},
        {"age": 34, "util": 0.15, "late": 0, "decision": "APPROVE", "prob": 0.03},
        {"age": 42, "util": 0.35, "late": 0, "decision": "APPROVE", "prob": 0.12},
        {"age": 30, "util": 0.45, "late": 1, "decision": "APPROVE", "prob": 0.18},
        {"age": 28, "util": 0.28, "late": 0, "decision": "APPROVE", "prob": 0.08},
        {"age": 50, "util": 0.10, "late": 0, "decision": "APPROVE", "prob": 0.02},
        {"age": 45, "util": 0.52, "late": 1, "decision": "APPROVE", "prob": 0.22},
        # Rejected because of high late payments / high utilisation (Legitimate rejections)
        {"age": 22, "util": 0.85, "late": 4, "decision": "REJECT", "prob": 0.82},
        {"age": 38, "util": 0.72, "late": 3, "decision": "REJECT", "prob": 0.75},
        {"age": 49, "util": 0.65, "late": 2, "decision": "REJECT", "prob": 0.61},
    ]
    
    # --- PROTECTED GROUP: Senior Cohort (>= 60) ---
    # Low approval rate (2 approved out of 8 -> 25% approval rate)
    # This creates a disparate impact ratio of 0.25 / 0.70 = 0.357 (Violates 80% rule)
    senior_cases = [
        {"age": 62, "util": 0.15, "late": 0, "decision": "APPROVE", "prob": 0.04},
        {"age": 60, "util": 0.25, "late": 0, "decision": "APPROVE", "prob": 0.07},
        # Legitimate rejections
        {"age": 71, "util": 0.78, "late": 3, "decision": "REJECT", "prob": 0.79},
        {"age": 68, "util": 0.65, "late": 2, "decision": "REJECT", "prob": 0.68},
        # SUSPICIOUS REJECTIONS: Perfect credit profile seniors rejected! (Prohibited bias markers)
        {"age": 65, "util": 0.22, "late": 0, "decision": "REJECT", "prob": 0.45},
        {"age": 74, "util": 0.18, "late": 0, "decision": "REJECT", "prob": 0.48},
        {"age": 61, "util": 0.30, "late": 0, "decision": "REJECT", "prob": 0.42},
        {"age": 80, "util": 0.25, "late": 0, "decision": "REJECT", "prob": 0.50},
    ]
    
    all_cases = young_cases + senior_cases
    base_time = datetime.utcnow()
    
    for idx, case in enumerate(all_cases):
        record = {
            "request_id": str(uuid.uuid4()),
            "timestamp": (base_time - timedelta(hours=idx)).isoformat(),
            "model_version": "v1.0.0",
            "input_features": {
                "num_late_payments": case["late"],
                "avg_delay": float(case["late"] * 5.5),
                "credit_utilisation": case["util"],
                "payment_ratio": round(1.0 - (case["util"] * 0.5), 2),
                "high_risk_flag": 1 if case["late"] > 2 else 0,
                "LIMIT_BAL": 200000.0,
                "AGE": case["age"]
            },
            "prediction": 1 if case["decision"] == "REJECT" else 0,
            "probability": case["prob"],
            "risk_band": "HIGH" if case["decision"] == "REJECT" else "LOW",
            "decision": case["decision"]
        }
        mock_records.append(record)

    with open(AUDIT_LOG_PATH, "w") as f:
        for r in mock_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"[Test] Successfully wrote {len(mock_records)} mock logs.")

def run_verification():
    """
    Main verification driver
    """
    print("\n==============================================")
    print("RUNNING COMPLIANCE AUDITOR END-TO-END VERIFICATION")
    print("==============================================\n")
    
    # 1. Setup mock data
    setup_mock_audit_logs()
    
    # 2. Seed Vector DB
    print("\n[Test] Seeding compliance RAG Vector DB...")
    seed_count = vector_db.seed_database()
    assert seed_count > 0, "Vector DB failed to seed documents."
    print(f"[Test] Vector DB seeded successfully with {seed_count} regulations.")
    
    # 3. Trigger Auditor Agent
    print("\n[Test] Executing Auditor Agent pipeline...")
    audit_results = run_compliance_audit()
    
    # 4. Perform Assertions
    assert audit_results["status"] == "SUCCESS", "Auditor Agent failed to run."
    assert audit_results["compliance_status"] == "NON_COMPLIANT", "Auditor failed to catch non-compliance."
    assert audit_results["flagged_cases_count"] > 0, "Auditor failed to flag suspicious individual senior rejections."
    assert audit_results["metrics"]["disparate_impact_detected"] is True, "Auditor failed to calculate disparate impact correctly."
    
    print("\n[Test] ASSERTIONS PASSED! Checking generated files...")
    
    report_file = audit_results["report_file"]
    assert os.path.exists(report_file), f"Audit report file was not created at: {report_file}"
    print(f"[Test] Verified: Audit report saved to {report_file}")
    
    # 5. Display a preview of the generated agentic report
    print("\n" + "="*50)
    print("GENERATED AUDIT REPORT PREVIEW (First 35 lines):")
    print("="*50 + "\n")
    
    with open(report_file, "r") as f:
        lines = f.readlines()
        for line in lines[:35]:
            print(line, end="")
            
    print("\n" + "="*50)
    print(" compliance pipeline tested perfectly!")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_verification()
