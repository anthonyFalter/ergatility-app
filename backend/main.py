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

# Initialize model handler
try:
    model_handler = ModelHandler(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize model handler: {str(e)}")
    model_handler = None
    
@app.get("/", tags=["Root"])
def read_root():
    '''Welcome endpoint'''
    return {
        "message": "Welcome to Employee Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
    
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    '''Check API and model health status'''
    return HealthCheckResponse(
        status="healthy"
        if model_handler and model_handler.is_loaded
        else "unhealthy",
        model_loaded=model_handler is not None and model_handler.is_loaded,
    )
    
@app.post("/predict", response_model=ChurnPredictionOutput, tags=["Prediction"])
def predict(input_data: ChurnInputSchema):
    '''Make a prediction based on employee metrics.'''
    if not model_handler or not model_handler.is_loaded:
        raise HTTPException(
            status_code=503, detail='Model not loaded. Service unavailable'
        )
        
        try:
            prediction, probability, risk_level = model_handler.predict(input_data)
            
            return ChurnPredictionOutput(
                prediction=prediction,
                probability=probability,
                risk_level=risk_level
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        