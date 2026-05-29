import json
import urllib.request
import urllib.error
import sys

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def query_running_compliance_api():
    print("==============================================")
    print("QUERYING ACTIVE COMPLIANCE API")
    print("==============================================\n")

    try:
        # 1. Seed the compliance vector database
        print("[Client] 1. Calling POST /compliance/seed...")
        seed_req = urllib.request.Request(
            "http://127.0.0.1:8000/compliance/seed", 
            method="POST"
        )
        with urllib.request.urlopen(seed_req) as response:
            seed_res = json.loads(response.read().decode())
            print("[Response]:")
            print(json.dumps(seed_res, indent=2))

        # 2. Trigger the agentic compliance audit
        print("\n[Client] 2. Calling POST /compliance/audit...")
        audit_req = urllib.request.Request(
            "http://127.0.0.1:8000/compliance/audit", 
            method="POST"
        )
        with urllib.request.urlopen(audit_req) as response:
            audit_res = json.loads(response.read().decode())
            print("[Response Summary]:")
            print(f"  - Status: {audit_res.get('status')}")
            print(f"  - Audit Summary: {audit_res.get('audit_summary')}")
            print(f"  - Compliance Status: {audit_res.get('compliance_status')}")
            print(f"  - Flagged Cases Count: {audit_res.get('flagged_cases_count')}")
            print(f"  - Saved Report Path: {audit_res.get('report_file')}")
            print("\n[Generated Audit Report Preview]:")
            
            # Print the report markdown returned by the API
            report_lines = audit_res.get('report_markdown', '').split('\n')
            for line in report_lines[:30]:
                print(f"    {line}")
                
    except urllib.error.URLError as e:
        print(f"[Client] API Call failed: {e}")
            
    print("\n==============================================")
    print("QUERY COMPLETE")
    print("==============================================\n")

if __name__ == "__main__":
    query_running_compliance_api()
