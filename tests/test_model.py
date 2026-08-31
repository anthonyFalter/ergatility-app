import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_handler import ModelHandler
from backend.config import MODEL_PATH, PREPROCESSOR_PATH


class TestModelHandler:
    """Tests for ModelHandler class"""
    
    @pytest.fixture
    def model_handler(self):
        """Create a ModelHandler instance"""
        return ModelHandler(MODEL_PATH, PREPROCESSOR_PATH)
    
    def test_model_initialization(self, model_handler):
        """Test that model initializes correctly"""
        assert model_handler.is_loaded, "Model should be loaded on initialization"
        assert model_handler.model is not None, "Model should not be None"
        assert model_handler.preprocessor is not None, "Preprocessor should not be None"
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
        assert PREPROCESSOR_PATH.exists(), f"Preprocessor file not found at {PREPROCESSOR_PATH}"
    
    def test_predict_output_shape(self, model_handler):
        """Test that prediction returns expected output"""
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        prediction, probability, confidence = model_handler.predict(features)
        
        assert isinstance(prediction, float), "Prediction should be float"
        assert isinstance(confidence, float), "Confidence should be float"
        assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"
        
        if probability is not None:
            assert isinstance(probability, list), "Probability should be list"
    
    def test_predict_with_different_features(self, model_handler):
        """Test prediction with various feature sets"""
        test_cases = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ]
        
        for features in test_cases:
            prediction, probability, confidence = model_handler.predict(features)
            assert isinstance(prediction, float), f"Failed for features: {features}"
            assert isinstance(confidence, float), f"Failed for features: {features}"
    
    def test_model_consistency(self, model_handler):
        """Test that same features produce same prediction"""
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        pred1, prob1, conf1 = model_handler.predict(features)
        pred2, prob2, conf2 = model_handler.predict(features)
        
        assert pred1 == pred2, "Predictions should be consistent"
        assert conf1 == conf2, "Confidence should be consistent"
    
    def test_preprocess_output_type(self, model_handler):
        """Test that preprocessing returns correct type"""
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        processed = model_handler.preprocess(features)
        
        assert isinstance(processed, list), "Preprocessed output should be list"
        assert len(processed) == len(features), "Should maintain feature count"


class TestModelLoadingErrors:
    """Tests for error handling in model loading"""
    
    def test_missing_model_file(self, tmp_path):
        """Test error when model file is missing"""
        missing_path = tmp_path / "missing_model.pickle"
        valid_path = PREPROCESSOR_PATH
        
        handler = ModelHandler(missing_path, valid_path)
        assert not handler.is_loaded, "Should not be loaded with missing model file"
    
    def test_missing_preprocessor_file(self, tmp_path):
        """Test error when preprocessor file is missing"""
        valid_path = MODEL_PATH
        missing_path = tmp_path / "missing_preprocessor.pickle"
        
        handler = ModelHandler(valid_path, missing_path)
        assert not handler.is_loaded, "Should not be loaded with missing preprocessor file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
