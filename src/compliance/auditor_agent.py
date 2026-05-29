import os
from datetime import datetime
from typing import Dict, Any, List
from src.compliance.config import COMPLIANCE_REPORT_DIR
from src.compliance.bias_profiler import compute_disparate_impact, flag_suspicious_cases
from src.compliance.rag_service import vector_db
from src.compliance.llm_client import generate_completion

def run_compliance_audit() -> Dict[str, Any]:
    """
    Executes the entire agentic auditing pipeline:
    1. Statistical Bias Profiler runs statistical analysis on logs.
    2. Identifies specific suspicious credit decisions.
    3. RAG pulls relevant legal clauses from the Regulation B Vector DB.
    4. CCO Agent (LLM) synthesizes findings and writes a formal Compliance Audit Report.
    """
    print("[Auditor Agent] Starting compliance audit pipeline...")
    
    # Step 1: Compute statistical bias metrics
    bias_metrics = compute_disparate_impact()
    
    # Step 2: Flag suspicious individual cases
    suspicious_cases = flag_suspicious_cases(limit=2)
    
    # Check if there is enough data or logs to perform an audit
    if bias_metrics["status"] == "NO_LOGS":
        return {
            "status": "COMPLETED_WITH_WARNINGS",
            "audit_summary": "Audit skipped: No decision logs found.",
            "metrics": bias_metrics,
            "report_markdown": "### No Audit Report Generated\nPlease make some credit predictions first to generate decision logs."
        }
    
    if bias_metrics["status"] == "INSUFFICIENT_DATA":
        return {
            "status": "COMPLETED_WITH_WARNINGS",
            "audit_summary": f"Audit skipped: Insufficient data ({bias_metrics['total_records']} records).",
            "metrics": bias_metrics,
            "report_markdown": "### No Audit Report Generated\nRequires a minimum of 5 logs to run statistical evaluations."
        }

    # Step 3: Run RAG retrieval to get applicable legal citations
    retrieved_regulations = []
    rag_citations_text = ""
    
    # Query for general discrimination if bias is detected
    if bias_metrics.get("disparate_impact_detected", False):
        rules = vector_db.search_rules("disparate impact age discrimination credit selection rate", limit=2)
        retrieved_regulations.extend(rules)
        
    # Query for individual cases
    for case in suspicious_cases:
        query = f"rejection age creditworthiness {case['reason_flagged']}"
        rules = vector_db.search_rules(query, limit=1)
        retrieved_regulations.extend(rules)

    # De-duplicate retrieved rules
    unique_rules = {rule["id"]: rule for rule in retrieved_regulations}.values()
    
    # Format rules for prompt
    for idx, rule in enumerate(unique_rules):
        rag_citations_text += f"\n### Citation {idx+1}: {rule['title']}\nCategory: {rule['category']}\nText: {rule['content']}\n"

    # Step 4: Construct prompts for the Chief Compliance Officer LLM Agent
    system_prompt = (
        "You are the Chief Compliance Officer at an audit-ready financial technology firm. "
        "Your role is to rigorously inspect ML model decisions and logs to ensure compliance with the "
        "Equal Credit Opportunity Act (ECOA - Regulation B). You are expert, detailed, objective, and authoritative."
    )
    
    user_prompt = f"""# REQUEST FOR COMPLIANCE AUDIT & FINDINGS REPORT

## 1. Statistical Log Metrics (Bias Profiler Output)
* **Total Ingested Log Records:** {bias_metrics.get('total_records')}
* **Disparate Impact Ratio (Age):** {bias_metrics.get('disparate_impact_ratio')}
* **Disparate Impact Detected:** {bias_metrics.get('disparate_impact_detected')}
* **Threshold Permitted:** {bias_metrics.get('threshold')}
* **Approval Rate (Protected Age Cohort 60+):** {bias_metrics.get('protected_approval_rate')}
* **Approval Rate (Reference Age Cohort <60):** {bias_metrics.get('reference_approval_rate')}

## 2. Suspicious Individual Decisions Flagged
"""
    
    if suspicious_cases:
        for idx, case in enumerate(suspicious_cases):
            user_prompt += f"""
### Case File #{idx+1}: Request ID: {case['request_id']}
* **Timestamp:** {case['timestamp']}
* **Model Predicted Probability of Default:** {case['probability_of_default']}
* **Decision:** {case['decision']}
* **Applicant Features:** {case['features']}
* **Trigger Flag:** {case['reason_flagged']}
"""
    else:
        user_prompt += "\nNo specific high-credit-quality individual rejections flagged as outliers.\n"
        
    user_prompt += f"""
## 3. Applicable Regulatory Guidelines (RAG Vector Store Citations)
{rag_citations_text}

---
## INSTRUCTION FOR THE REPORT
Based on the data above, generate a highly professional, beautifully formatted, audit-ready **Compliance Audit Report**. 
Ensure you:
1. State the overall compliance status clearly (e.g., COMPLIANT, WARNING, or NON-COMPLIANT).
2. Connect the **Statistical Log Metrics** and **Suspicious Cases** directly to the **Regulatory Citations**. Explain precisely which sections were violated or are at risk.
3. If an individual case is flagged, explain why rejecting an applicant with low utilisation and zero late payments is a potential ECOA violation when their age is senior.
4. Give concrete, actionable remediation steps for the engineering and model development teams (e.g., threshold adjustments, feature restrictions, adversarial retraining).
5. Output the response in Markdown format.
"""

    # Step 5: Invoke LLM (with fallback support) to generate report
    report_content = generate_completion(system_prompt, user_prompt)
    
    # Save the report to disk
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_filename = f"audit_report_{timestamp}.md"
    report_filepath = COMPLIANCE_REPORT_DIR / report_filename
    
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"[Auditor Agent] Successfully saved audit report to {report_filepath}")
    
    audit_summary_status = "NON_COMPLIANT" if bias_metrics.get("disparate_impact_detected", False) or len(suspicious_cases) > 0 else "COMPLIANT"
    
    return {
        "status": "SUCCESS",
        "audit_summary": f"Audit completed. Status: {audit_summary_status}",
        "compliance_status": audit_summary_status,
        "metrics": bias_metrics,
        "flagged_cases_count": len(suspicious_cases),
        "report_file": str(report_filepath),
        "report_markdown": report_content
    }
