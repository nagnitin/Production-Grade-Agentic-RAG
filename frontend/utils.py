"""
API utilities for the Streamlit frontends.

WHY: Provides a unified interface for the frontend to communicate with the FastAPI backend.
Reads configuration from environment variables and handles authentication, request/response lifecycle,
and error handling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import requests

# Base configuration from environment variables
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api/v1")
API_KEY = os.environ.get("API_KEY", "changeme")


def get_headers() -> Dict[str, str]:
    """Build the request headers with API key authentication."""
    return {
        "x-api-key": API_KEY,
    }


def get_health_status() -> Dict[str, Any]:
    """Retrieve backend system health status."""
    try:
        response = requests.get(
            f"{os.path.dirname(BACKEND_URL)}/health",  # /health is root-level, not /api/v1
            headers=get_headers(),
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "message": f"Status code {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def query_rag(
    query: str,
    session_id: str,
    user_id: str = "streamlit_user",
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a query to the Agentic RAG pipeline."""
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "filters": filters or {},
    }
    
    try:
        response = requests.post(url, json=payload, headers=get_headers(), timeout=60)
        if response.status_code == 200:
            return response.json()
        
        # Handle specific error responses
        try:
            err_detail = response.json().get("detail", response.text)
        except Exception:
            err_detail = response.text
            
        return {"error": f"API Error ({response.status_code}): {err_detail}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}


def upload_document(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Upload and ingest a document into the system."""
    url = f"{BACKEND_URL}/ingest"
    files = {"file": (filename, file_content, "application/octet-stream")}
    
    try:
        response = requests.post(url, files=files, headers=get_headers(), timeout=120)
        if response.status_code == 200:
            return response.json()
        
        try:
            err_detail = response.json().get("detail", response.text)
        except Exception:
            err_detail = response.text
        return {"error": f"Ingestion Error ({response.status_code}): {err_detail}"}
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


def submit_feedback(
    session_id: str,
    rating: int,
    comment: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit user feedback for an answer."""
    url = f"{BACKEND_URL}/feedback"
    payload = {
        "session_id": session_id,
        "rating": rating,
        "comment": comment,
        "message_id": message_id,
    }
    
    try:
        response = requests.post(url, json=payload, headers=get_headers(), timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Feedback submission failed: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def trigger_evaluation(dataset_name: str, metrics: List[str]) -> Dict[str, Any]:
    """Trigger a RAGAS evaluation run against a golden dataset."""
    url = f"{BACKEND_URL}/evaluate"
    payload = {
        "dataset_name": dataset_name,
        "metrics": metrics,
    }
    
    try:
        response = requests.post(url, json=payload, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to trigger evaluation: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_evaluation_runs() -> List[Dict[str, Any]]:
    """Retrieve all past and active evaluation runs."""
    url = f"{BACKEND_URL}/evaluate/runs"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def get_evaluation_run_details(run_id: str) -> Dict[str, Any]:
    """Retrieve details and sample-level metrics for a specific run."""
    url = f"{BACKEND_URL}/evaluate/runs/{run_id}"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to fetch run details: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_system_metrics() -> Dict[str, Any]:
    """Retrieve operational telemetry and metrics."""
    url = f"{BACKEND_URL}/metrics"
    try:
        response = requests.get(url, headers=get_headers(), timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}


def clear_documents() -> Dict[str, Any]:
    """Delete all points and documents in the vector store and clear cache."""
    url = f"{BACKEND_URL}/ingest/clear"
    try:
        response = requests.post(url, headers=get_headers(), timeout=15)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to clear: {response.text}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

