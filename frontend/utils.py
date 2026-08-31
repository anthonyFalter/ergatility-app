import requests
import streamlit as st
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with the backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoints = {
            "health": f"{base_url}/health",
            "predict": f"{base_url}/predict",
            "batch_predict": f"{base_url}/batch-predict"
        }
    
    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if API is healthy"""
        try:
            response = requests.get(self.endpoints["health"], timeout=5)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False, {"error": str(e)}
    
    def predict(self, features: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """Make a single prediction"""
        try:
            payload = {"features": features}
            response = requests.post(self.endpoints["predict"], json=payload, timeout=30)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to API")
            return False, {"error": "Cannot connect to API server"}
        except Exception as e:
            logger.error(f"Prediction request failed: {str(e)}")
            return False, {"error": str(e)}
    
    def batch_predict(self, features_list: List[List[float]]) -> Tuple[bool, Dict[str, Any]]:
        """Make batch predictions"""
        try:
            payload = [{"features": features} for features in features_list]
            response = requests.post(self.endpoints["batch_predict"], json=payload, timeout=30)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to API")
            return False, {"error": "Cannot connect to API server"}
        except Exception as e:
            logger.error(f"Batch prediction request failed: {str(e)}")
            return False, {"error": str(e)}


def format_prediction_display(result: Dict[str, Any]) -> None:
    """Format and display prediction results in Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", f"{result['prediction']:.2f}")
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.2%}")
    
    with col3:
        if result.get('probability'):
            st.metric("Max Probability", f"{max(result['probability']):.2%}")


def display_error(message: str) -> None:
    """Display error message in Streamlit"""
    st.error(f"Error: {message}")


def display_warning(message: str) -> None:
    """Display warning message in Streamlit"""
    st.warning(f"Warning: {message}")


def display_success(message: str) -> None:
    """Display success message in Streamlit"""
    st.success(f"Success: {message}")
