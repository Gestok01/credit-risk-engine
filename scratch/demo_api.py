import time
import json
import urllib.request
import urllib.error
import subprocess
import sys
import os
from pathlib import Path

def run_live_api_demo():
    print("==============================================")
    print("STARTING LIVE COMPLIANCE API DEMO")
    print("==============================================\n")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    stdout_file = open(logs_dir / "demo_uvicorn_out.log", "w")
    stderr_file = open(logs_dir / "demo_uvicorn_err.log", "w")

    # 1. Start the FastAPI server in the background
    print("[Demo] Starting Uvicorn server on port 8000...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--port", "8000", "--host", "127.0.0.1"],
        stdout=stdout_file,
        stderr=stderr_file
    )
    
    # Give the server a moment to spin up
    time.sleep(4.0)
    
    # Check if server started successfully
    if server_process.poll() is not None:
        print("[Demo] Error: Uvicorn server failed to start immediately.")
        stdout_file.close()
        stderr_file.close()
        with open(logs_dir / "demo_uvicorn_err.log", "r") as err_log:
            print("[Server Stderr]:\n", err_log.read())
        return

    print("[Demo] Uvicorn server process is running!")

    try:
        # 2. Seed the compliance vector database
        print("\n[Demo] 1. Calling POST /compliance/seed...")
        seed_req = urllib.request.Request(
            "http://127.0.0.1:8000/compliance/seed", 
            method="POST"
        )
        with urllib.request.urlopen(seed_req) as response:
            seed_res = json.loads(response.read().decode())
            print("[Response]:")
            print(json.dumps(seed_res, indent=2))

        # 3. Trigger the agentic compliance audit
        print("\n[Demo] 2. Calling POST /compliance/audit...")
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
            
            # Print first 20 lines of the report markdown returned by the API
            report_lines = audit_res.get('report_markdown', '').split('\n')
            for line in report_lines[:25]:
                print(f"    {line}")
                
    except urllib.error.URLError as e:
        print(f"[Demo] API Call failed: {e}")
        # Wait a moment for logs to write, then terminate process and read stderr
        server_process.terminate()
        server_process.wait(timeout=2)
        stdout_file.close()
        stderr_file.close()
        with open(logs_dir / "demo_uvicorn_err.log", "r") as err_log:
            print("[Server Stderr]:\n", err_log.read())
        return
        
    finally:
        # 4. Clean up: Shutdown the Uvicorn server
        if server_process.poll() is None:
            print("\n[Demo] Shutting down Uvicorn server process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("[Demo] Server process terminated cleanly.")
            except subprocess.TimeoutExpired:
                server_process.kill()
                print("[Demo] Server process force-killed.")
        
        stdout_file.close()
        stderr_file.close()
            
    print("\n==============================================")
    print("LIVE COMPLIANCE API DEMO COMPLETE")
    print("==============================================\n")

if __name__ == "__main__":
    run_live_api_demo()
