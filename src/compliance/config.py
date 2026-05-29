import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Storage Paths
CHROMA_DB_DIR = str(DATA_DIR / "compliance_chroma")
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", str(LOGS_DIR / "audit_logs.jsonl"))
COMPLIANCE_REPORT_DIR = LOGS_DIR / "compliance_reports"
COMPLIANCE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Compliance Thresholds
DISPARATE_IMPACT_THRESHOLD = 0.8  # The standard 80% rule for adverse impact
PROTECTED_AGE_THRESHOLD = 60      # Age 60+ considered senior/protected class
MIN_SAMPLES_FOR_BIAS = 5          # Minimum samples in each group before flagging statistical bias
