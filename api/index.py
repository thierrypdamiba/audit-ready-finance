"""Vercel serverless entry point for FastAPI app."""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set env before any imports
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

from app import app  # noqa: E402
