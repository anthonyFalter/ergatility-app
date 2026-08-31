import os
import requests

# Base URL for API calls
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def check_api_health() -> bool:
    '''Verify FastAPI backend and model healthiness'''
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