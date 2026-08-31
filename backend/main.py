import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ALLOWED_ORIGINS, MODEL_PATH, PREPROCESSOR_PATH
from backend.model_handler import ModelHandler
from backend.schemas import (
    ChurnInputSchemma,
    ChurnPredictionOutput,
    HealthCheckResponse,
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title = "Ergatility Employee Turnover Prediction API",
    description="REST API for predicting employee churn risk",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)