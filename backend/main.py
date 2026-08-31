from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.config import ALLOWED_ORIGINS, MODEL_PATH, PREPROCESSOR_PATH
from backend.schemas import PredictionInput, PredictionOutput, HealthCheck
from backend.model_handler import ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ergatility Prediction API",
    description="ML model inference API for predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model handler
try:
    model_handler = ModelHandler(MODEL_PATH, PREPROCESSOR_PATH)
except Exception as e:
    logger.error(f"Failed to initialize model handler: {str(e)}")
    model_handler = None


@app.get("/", tags=["Root"])
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Ergatility Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
def health_check():
    """Check API and model health status"""
    return HealthCheck(
        status="healthy" if model_handler and model_handler.is_loaded else "unhealthy",
        model_loaded=model_handler is not None and model_handler.is_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Make a prediction based on input features
    
    Input:
        - features: List of numeric values
    
    Returns:
        - prediction: Model prediction
        - probability: Class probabilities (if available)
        - confidence: Confidence score of the prediction
    """
    if not model_handler or not model_handler.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        prediction, probability, confidence = model_handler.predict(input_data.features)
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(batch_inputs: list[PredictionInput]):
    """
    Make batch predictions
    
    Input:
        - List of prediction inputs
    
    Returns:
        - List of prediction outputs
    """
    if not model_handler or not model_handler.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        results = []
        for input_data in batch_inputs:
            prediction, probability, confidence = model_handler.predict(input_data.features)
            results.append({
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence
            })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT, API_RELOAD
    
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
