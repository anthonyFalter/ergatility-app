from pathlib import Path
import sys
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MODEL_PATH
from backend.model_handler import ModelHandler


class TestModelHandler:
    """Tests for ModelHandler class."""

    @pytest.fixture
    def model_handler(self):
        """Create a ModelHandler instance."""
        return ModelHandler(MODEL_PATH)

    @pytest.fixture
    def sample_input_dict(self):
        """Sample employee dictionary input matching API schema."""
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

    def test_model_initialization(self, model_handler):
        """Test that model initializes correctly."""
        assert (
            model_handler.is_loaded
        ), "Model should be loaded on initialization"
        assert model_handler.model is not None, "Model should not be None"

    def test_model_files_exist(self):
        """Test that model artifact exists."""
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

    def test_predict_output_types_and_bounds(
        self, model_handler, sample_input_dict
    ):
        """Test that prediction returns expected binary target and probability bounds."""
        prediction, probability = model_handler.predict(sample_input_dict)

        assert isinstance(prediction, (int, float))
        assert prediction in [0, 1], "Prediction target should be 0 or 1"

        assert isinstance(
            probability, float
        ), "Probability should be a float"
        assert (
            0.0 <= probability <= 1.0
        ), "Probability should be between 0 and 1"

    def test_model_consistency(self, model_handler, sample_input_dict):
        """Test that identical input features produce deterministic output."""
        pred1, prob1 = model_handler.predict(sample_input_dict)
        pred2, prob2 = model_handler.predict(sample_input_dict)

        assert pred1 == pred2, "Predictions should be consistent"
        assert prob1 == prob2, "Probabilities should be consistent"

    def test_preprocess_output_format(self, model_handler, sample_input_dict):
        """Test that preprocessing transforms raw dict input into a pandas DataFrame."""
        processed_df = model_handler.preprocess(sample_input_dict)

        assert isinstance(
            processed_df, pd.DataFrame
        ), "Preprocessed feature structure must be a DataFrame"
        assert (
            not processed_df.empty
        ), "Preprocessed DataFrame should not be empty"


class TestModelLoadingErrors:
    """Tests for error handling in model loading."""

    def test_missing_model_file(self, tmp_path):
        """Test error handling when model file path does not exist."""
        missing_path = tmp_path / "non_existent_model.pickle"
        handler = ModelHandler(missing_path)

        assert (
            not handler.is_loaded
        ), "ModelHandler should set is_loaded=False when path is invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])