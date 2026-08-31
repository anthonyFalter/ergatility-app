from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionInput(BaseModel):
    """Schema for model input data"""
    features: List[float] = Field(..., description="List of input features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }


class PredictionOutput(BaseModel):
    """Schema for model prediction output"""
    prediction: float = Field(..., description="Model prediction")
    probability: Optional[List[float]] = Field(None, description="Prediction probabilities")
    confidence: float = Field(..., description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1.0,
                "probability": [0.1, 0.9],
                "confidence": 0.95
            }
        }


class HealthCheck(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(default="1.0.0", description="API version")
