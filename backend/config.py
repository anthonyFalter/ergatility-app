import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model artifacts paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "rf_cv_model.pickle"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pickle"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

# Model Configuration
MODEL_TIMEOUT = 30  # seconds

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
]
