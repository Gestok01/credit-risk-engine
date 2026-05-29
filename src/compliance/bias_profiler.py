import json
import os
import pandas as pd
from typing import Dict, Any, List
from src.compliance.config import AUDIT_LOG_PATH, DISPARATE_IMPACT_THRESHOLD, PROTECTED_AGE_THRESHOLD, MIN_SAMPLES_FOR_BIAS

def compute_disparate_impact() -> Dict[str, Any]:
    """
    Reads the decision logs and computes statistical bias metrics (Disparate Impact Ratio)
    for applicants based on age.
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return {
            "status": "NO_LOGS",
            "message": "No decision logs found to analyze.",
            "disparate_impact_ratio": 1.0,
            "total_records": 0
        }

    records = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Ensure it is a prediction record and not a retraining event
                if "input_features" in data and "AGE" in data["input_features"]:
                    records.append({
                        "request_id": data.get("request_id"),
                        "timestamp": data.get("timestamp"),
                        "age": data["input_features"]["AGE"],
                        "credit_utilisation": data["input_features"].get("credit_utilisation", 0.0),
                        "num_late_payments": data["input_features"].get("num_late_payments", 0),
                        "decision": data.get("decision", "REVIEW")
                    })
            except json.JSONDecodeError:
                continue

    if len(records) < MIN_SAMPLES_FOR_BIAS:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Insufficient records ({len(records)}/{MIN_SAMPLES_FOR_BIAS}) to compute statistical bias.",
            "disparate_impact_ratio": 1.0,
            "total_records": len(records)
        }

    df = pd.DataFrame(records)
    df["is_protected"] = df["age"] >= PROTECTED_AGE_THRESHOLD
    df["is_approved"] = df["decision"] != "REJECT"  # Non-rejection is the selection criteria

    # Calculate selection rates
    protected_group = df[df["is_protected"]]
    reference_group = df[~df["is_protected"]]

    if len(protected_group) == 0 or len(reference_group) == 0:
        return {
            "status": "SINGLE_GROUP_DATA",
            "message": "Logs only contain one age group. Cannot calculate comparative bias.",
            "disparate_impact_ratio": 1.0,
            "total_records": len(df)
        }

    protected_approved = protected_group["is_approved"].sum()
    reference_approved = reference_group["is_approved"].sum()

    protected_selection_rate = protected_approved / len(protected_group)
    reference_selection_rate = reference_approved / len(reference_group)

    # Prevent division by zero
    if reference_selection_rate == 0:
        dir_ratio = 1.0
    else:
        dir_ratio = protected_selection_rate / reference_selection_rate

    # Check if ratio violates the 80% rule
    is_biased = dir_ratio < DISPARATE_IMPACT_THRESHOLD

    return {
        "status": "ANALYZED",
        "total_records": len(df),
        "protected_class_size": len(protected_group),
        "reference_class_size": len(reference_group),
        "protected_approval_rate": round(protected_selection_rate, 4),
        "reference_approval_rate": round(reference_selection_rate, 4),
        "disparate_impact_ratio": round(dir_ratio, 4),
        "disparate_impact_detected": bool(is_biased),
        "threshold": DISPARATE_IMPACT_THRESHOLD,
        "message": (
            f"Disparate impact ratio is {round(dir_ratio, 2)}. "
            f"Approval rate for age {PROTECTED_AGE_THRESHOLD}+ is {round(protected_selection_rate * 100, 1)}% "
            f"vs {round(reference_selection_rate * 100, 1)}% for younger cohorts."
        )
    }

def flag_suspicious_cases(limit: int = 3) -> List[Dict[str, Any]]:
    """
    Identifies specific individual records that look statistically suspicious and
    warrant formal compliance review by the Agent.
    Suspicious senior cases: High-credit-quality older applicants who were rejected.
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    suspicious_records = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "input_features" in data and "AGE" in data["input_features"]:
                    features = data["input_features"]
                    decision = data.get("decision", "REVIEW")
                    
                    # Rule: Applicant is senior, rejected, but has low utilization (< 0.4) and no late payments
                    if (
                        features.get("AGE", 0) >= PROTECTED_AGE_THRESHOLD
                        and decision == "REJECT"
                        and features.get("credit_utilisation", 1.0) < 0.4
                        and features.get("num_late_payments", 99) == 0
                    ):
                        suspicious_records.append({
                            "request_id": data.get("request_id"),
                            "timestamp": data.get("timestamp"),
                            "probability_of_default": data.get("probability"),
                            "decision": decision,
                            "features": features,
                            "reason_flagged": (
                                f"Senior applicant (Age {features['AGE']}) was REJECTED despite strong credit markers: "
                                f"credit utilisation is {round(features['credit_utilisation'] * 100, 1)}% and 0 late payments."
                            )
                        })
            except json.JSONDecodeError:
                continue

    # Sort suspicious cases so that lowest probability of default (most creditworthy) comes first
    suspicious_records.sort(key=lambda x: x.get("probability_of_default", 1.0))
    return suspicious_records[:limit]
