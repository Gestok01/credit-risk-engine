Credit Risk Engine – Audit-Ready ML Inference Service
A production-oriented Credit Risk Prediction API built to demonstrate how machine learning models are served, explained, monitored, and audited in real-world systems — not just trained in notebooks.
This project focuses on the engineering side of ML, covering inference pipelines, explainability, drift monitoring, containerization, and cloud deployment.

🔍 Why this project exists
Most ML projects stop at “trained a model, got 92% accuracy.”
This one starts after that point.
The goal was to answer:


How is a trained model served safely?


How do we explain its decisions?


How do we detect when it starts lying (data drift)?


How do we make it auditable and deployment-ready?



🧠 Model Overview


Type: Binary classification (default risk)


Algorithm: Logistic Regression (scikit-learn pipeline)


Target: Probability of default


Output: Risk band + decision


The trained pipeline is serialized and loaded directly at runtime.

🏗️ System Architecture
Client
  │
  │ JSON request
  ▼
FastAPI Service
  ├── Input validation (Pydantic)
  ├── Model inference
  ├── Risk band mapping
  ├── Decision rules
  ├── SHAP explainability
  ├── Drift monitoring
  │     │
  │     └── Audit logs ──► [Agentic Compliance Auditor]
  │                              ├── Bias Profiler (disparate impact check)
  │                              ├── RAG Vector Store (ECOA Reg B search)
  │                              └── CCO Agent (LLM synthesis & reports)
  │
  └── Compliance Endpoints


🚀 Live Deployment


Swagger UI:
👉 http://3.106.188.194:8000/docs


Health Check:
GET /



📦 API Endpoints
1️⃣ Predict Credit Risk
Endpoint
POST /predict

Request Payload (actual implementation)
{
  "num_late_payments": 2,
  "avg_delay": 12.5,
  "credit_utilisation": 0.42,
  "payment_ratio": 0.78,
  "high_risk_flag": 0,
  "LIMIT_BAL": 250000,
  "AGE": 34
}

Response
{
  "default_prediction": 0,
  "probability_of_default": 0.27,
  "risk_band": "MEDIUM",
  "decision": "REVIEW"
}


2️⃣ Explain Prediction (SHAP)
Endpoint
POST /explain

Uses the same feature vector as inference and returns SHAP-based feature attributions explaining why the model predicted a given risk.
✔ Ensures prediction–explanation consistency

3️⃣ Monitor Data Drift
Endpoint
POST /monitor/drift

Purpose


Compares incoming feature distributions against reference data


Detects statistically significant drift


Flags retraining recommendations


⚠️ Retraining is not auto-triggered by design — human approval is required.

4️⃣ Retraining Readiness Check
Endpoint
POST /monitor/retraining-check

Returns a signal indicating whether recent drift patterns suggest that retraining should be considered.

5️⃣ Seed Compliance Vector DB
Endpoint
POST /compliance/seed

Seeds the local vector database with federal regulations (Equal Credit Opportunity Act Regulation B).

6️⃣ Agentic Compliance Audit
Endpoint
POST /compliance/audit

Runs the multi-agent AI compliance auditor over decision logs to detect systemic bias and flag individual outlier senior rejections. Returns statistical metrics and an audit-ready compliance markdown report.

🧾 Audit & Traceability


Inference inputs and decisions are logged


Risk band mapping and decision rules are deterministic


Enables post-hoc inspection for:


Compliance


Debugging


Model behavior review




This mirrors real regulated ML systems (finance / risk).

🐳 Containerization


Dockerized FastAPI service


Clean separation of:


Application code


Model artifact


Dependencies




docker build -t credit-risk-engine .
docker run -p 8000:8000 credit-risk-engine


☁️ Cloud Deployment


Hosted on AWS EC2 (Amazon Linux 2023)


Exposed via port 8000


Swagger UI publicly accessible



🔁 CI/CD (Engineering Reality)
GitHub Actions was used to experiment with EC2-based deployment automation.
This surfaced real-world issues including:


SSH connectivity timeouts


Remote Docker build context problems


Repository path mismatches on EC2


These challenges were intentionally not hidden, as they reflect common production deployment pitfalls.

🛠️ Tech Stack


Python


FastAPI


scikit-learn


SHAP


ChromaDB & Custom TF-IDF (Vector DB)


OpenAI / Ollama (LLM Agents)


Docker


AWS EC2


GitHub Actions


Pydantic



📂 Repository Structure
credit-risk-engine/
│
├── src/
│   ├── compliance/       # Pluggable Agentic compliance package
│   │   ├── config.py     # Central compliance thresholds & configs
│   │   ├── rules_data.py # ECOA Regulation B statutory text
│   │   ├── rag_service.py # Primary ChromaDB + fallback vector store
│   │   ├── bias_profiler.py # Bias analyzer (80% rule & outlier filter)
│   │   ├── llm_client.py # Pluggable OpenAI/Ollama client
│   │   └── auditor_agent.py # Orchestrating Chief Compliance Officer agent
│   │
│   ├── model.py
│   ├── services/
│   ├── schemas.py
│   ├── config.py
│   └── main.py
│
├── scratch/              # Verification & demo scripts
│   ├── test_compliance.py # Automated end-to-end verification script
│   └── demo_api.py       # Live API demonstration script
│
├── notebooks/
├── data/
├── logs/                 # Active audit logs & generated compliance reports
├── Dockerfile
├── requirements.txt
├── credit_risk_pipeline.pkl
└── README.md


🎯 What this project demonstrates


Serving ML models as real APIs


Explainability with SHAP


Drift detection in production settings


Audit-ready ML design


Cloud deployment with Docker


Practical CI/CD lessons (not toy pipelines)



