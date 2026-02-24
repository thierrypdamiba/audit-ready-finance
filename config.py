import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Server
PORT = 6847
HOST = "0.0.0.0"

# Models
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"

# Qdrant
COLLECTION_NAME = "sec_filings"

# SEC Filings
TICKERS = ["AAPL", "TSLA", "JPM"]
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla, Inc.",
    "JPM": "JPMorgan Chase & Co.",
}

# Short aliases for query detection
COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "TSLA": ["tesla", "tsla"],
    "JPM": ["jpmorgan", "jp morgan", "jpm", "chase"],
}

# Chunking
CHUNK_SIZE = 500  # tokens (approx chars / 4)
CHUNK_OVERLAP = 50

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FILINGS_DIR = os.path.join(DATA_DIR, "filings")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Audit DB (use /tmp in serverless environments like Vercel)
_is_serverless = os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
AUDIT_DB_PATH = (
    "/tmp/audit_trail.db" if _is_serverless
    else os.path.join(os.path.dirname(__file__), "audit_trail.db")
)

# Sections to extract from 10-K
TARGET_SECTIONS = {
    "1": "business",
    "1A": "risk_factors",
    "7": "md_and_a",
    "9A": "controls",
}
