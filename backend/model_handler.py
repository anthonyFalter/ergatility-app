import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handler for loading and performing inference with the ML model"""
    
    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        """Load model and preprocessor from pickle files"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.is_loaded = True
            logger.info("Model and preprocessor loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def preprocess(self, features: List[float]) -> List[float]:
        """Preprocess input features using the loaded preprocessor"""
        if not self.is_loaded or self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")
        
        try:
            # If preprocessor is a sklearn scaler, it expects 2D array
            processed = self.preprocessor.transform([[f for f in features]])
            return processed[0].tolist()
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def predict(self, features: List[float]) -> Tuple[float, Optional[List[float]], float]:
        """
        Make prediction on input features
        
        Returns:
            Tuple of (prediction, probabilities, confidence)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess features
            processed_features = self.preprocess(features)
            
            # Make prediction
            prediction = self.model.predict([processed_features])[0]
            
            # Get probability scores if available
            probability = None
            confidence = 1.0
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([processed_features])[0]
                probability = proba.tolist()
                confidence = float(max(proba))
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function([processed_features])[0]
                confidence = float(abs(decision) / (1 + abs(decision)))
            
            return float(prediction), probability, confidence
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
