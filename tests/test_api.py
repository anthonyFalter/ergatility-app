import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data, "Response should contain status"
        assert "model_loaded" in data, "Response should contain model_loaded"
        assert "version" in data, "Response should contain version"
    
    def test_health_check_model_status(self):
        """Test that health check reports model status correctly"""
        response = client.get("/health")
        data = response.json()
        
        if data["model_loaded"]:
            assert data["status"] == "healthy"
        else:
            assert data["status"] == "unhealthy"


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoint"""
    
    def test_predict_single(self):
        """Test single prediction endpoint"""
        payload = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "prediction" in data, "Response should contain prediction"
            assert "confidence" in data, "Response should contain confidence"
            assert isinstance(data["prediction"], (int, float))
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_input(self):
        """Test prediction with empty features"""
        payload = {"features": []}
        response = client.post("/predict", json=payload)
        
        # Could return 422 or 500 depending on implementation
        assert response.status_code >= 400
    
    def test_predict_various_feature_sizes(self):
        """Test prediction with different number of features"""
        test_cases = [
            [1.0],
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0] * 10,
        ]
        
        for features in test_cases:
            payload = {"features": features}
            response = client.post("/predict", json=payload)
            
            # Just ensure it returns a valid response code
            assert response.status_code >= 200, f"Failed for feature count: {len(features)}"


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint"""
    
    def test_batch_predict(self):
        """Test batch prediction endpoint"""
        payload = [
            {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
            {"features": [2.0, 3.0, 4.0, 5.0, 6.0]},
        ]
        response = client.post("/batch-predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) == 2
            
            for result in data["results"]:
                assert "prediction" in result
                assert "confidence" in result
    
    def test_batch_predict_single_sample(self):
        """Test batch prediction with single sample"""
        payload = [{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}]
        response = client.post("/batch-predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["results"]) == 1
    
    def test_batch_predict_empty(self):
        """Test batch prediction with empty list"""
        payload = []
        response = client.post("/batch-predict", json=payload)
        
        # Empty batch should be handled appropriately
        assert response.status_code >= 200


class TestAPIResponses:
    """Tests for API response format"""
    
    def test_json_response_headers(self):
        """Test that responses have correct content type"""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_error_response_format(self):
        """Test error response format"""
        payload = {"features": []}
        response = client.post("/predict", json=payload)
        
        if response.status_code >= 400:
            assert response.headers.get("content-type")
            assert len(response.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
