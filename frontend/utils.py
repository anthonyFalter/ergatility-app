import os
import requests

# Base URL for API calls - defaults to local FastAPI server, falls back to Railway environment variable
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Verify FastAPI backend and model healthiness."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy" and data.get(
                "model_loaded", False
            )

        return False
    except requests.exceptions.RequestException:
        return False


def get_prediction(payload: dict) -> dict | None:
    """Send input features to backend API for prediction results."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API Error ({response.status_code}): {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection failed: {str(e)}"}