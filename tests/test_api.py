from pathlib import Path
import sys
from fastapi.testclient import TestClient
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test health check endpoint structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data, "Response should contain status"
        assert "model_loaded" in data, "Response should contain model_loaded"

    def test_health_check_model_status(self):
        """Test that health check reports status based on model_loaded."""
        response = client.get("/health")
        data = response.json()

        if data["model_loaded"]:
            assert data["status"] == "healthy"
        else:
            assert data["status"] == "unhealthy"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns welcome metadata."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestPredictionEndpoint:
    """Tests for churn prediction endpoint."""

    @pytest.fixture
    def valid_payload(self):
        """Valid sample payload based on ChurnInputSchema."""
        return {
            "satisfaction_level": 0.75,
            "last_evaluation": 0.78,
            "number_project": 4,
            "average_montly_hours": 200,
            "time_spend_company": 3,
            "work_accident": 0,
            "promotion_last_5years": 0,
            "salary": "medium",
            "department": "sales",
        }

    def test_predict_success(self, valid_payload):
        """Test prediction with a valid payload."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data, "Response should contain prediction"
        assert "probability" in data, "Response should contain probability"
        assert "risk_level" in data, "Response should contain risk_level"

        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["High Risk", "Low Risk"]

    def test_predict_invalid_boundary(self, valid_payload):
        """Test validation error when numeric constraints are violated."""
        # Out-of-bounds satisfaction_level (> 1.0)
        invalid_payload = valid_payload.copy()
        invalid_payload["satisfaction_level"] = 1.5

        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_enum_choice(self, valid_payload):
        """Test validation error when invalid enum literals are passed."""
        invalid_payload = valid_payload.copy()
        invalid_payload["salary"] = "very_high"
        invalid_payload["department"] = "invalid_dept"

        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_missing_required_fields(self):
        """Test validation error when sending an incomplete request."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestAPIResponses:
    """Tests for API response header & formatting."""

    def test_json_response_headers(self):
        """Test that responses return application/json headers."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_validation_error_format(self):
        """Test validation error response structure on bad requests."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])