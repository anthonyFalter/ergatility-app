# Ergatility Prediction System

A machine learning prediction system with a FastAPI backend and Streamlit frontend for interactive predictions using a pre-trained Random Forest model.

## Project Structure

```
ergatility-app/
├── artifacts/                      
│   ├── rf_cv_model.pickle          # Trained Random Forest model
│   └── preprocessor.pickle         # Data scaler/encoder
│
├── backend/                        
│   ├── main.py                     # FastAPI application & routes
│   ├── schemas.py                  # Pydantic data validation
│   ├── model_handler.py            # Model loading & inference
│   ├── config.py                   # Configuration & constants
│   └── __init__.py
│
├── frontend/                       
│   ├── app.py                      # Streamlit dashboard
│   ├── utils.py                    # API client & helpers
│   └── __init__.py
│
├── tests/                          
│   ├── test_api.py                 # API endpoint tests
│   └── test_model.py               # Model inference tests
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone or download the project:
```bash
cd ergatility-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Backend API

Start the FastAPI server:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

### Running the Frontend

In a separate terminal, start the Streamlit app:

```bash
streamlit run frontend/app.py
```

The frontend will open at `http://localhost:8501`

## API Endpoints

### Health Check
- **GET** `/health`
  - Returns API and model status
  - Response: `{status, model_loaded, version}`

### Single Prediction
- **POST** `/predict`
  - Input: `{features: [float, ...]}`
  - Response: `{prediction: float, probability: [float, ...], confidence: float}`

### Batch Predictions
- **POST** `/batch-predict`
  - Input: `[{features: [float, ...]}, ...]`
  - Response: `{results: [{prediction, probability, confidence}, ...]}`

## Example API Usage

### Using curl

Single prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

Batch predictions:
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '[{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}, {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}]'
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch-predict",
    json=[
        {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
        {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}
    ]
)
print(response.json())
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py
pytest tests/test_model.py

# Run with verbose output
pytest -v

# Run with coverage
pip install pytest-cov
pytest --cov=backend --cov=frontend
```

## Configuration

Edit `backend/config.py` to customize:
- API host and port
- CORS allowed origins
- Model file paths
- Model timeout

Set environment variables:
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export API_RELOAD=True
```

## Model Details

The system uses a scikit-learn Random Forest model with:
- Input: Feature vector (preprocessed)
- Output: Numeric prediction with confidence score
- Preprocessing: StandardScaler or similar (loaded from pickle)

## Features

- **FastAPI Backend**: High-performance async API with automatic documentation
- **Streamlit Frontend**: Interactive web UI for predictions
- **Batch Processing**: Support for processing multiple samples simultaneously
- **Health Monitoring**: API health and model status checks
- **Error Handling**: Comprehensive error messages and validation
- **CORS Support**: Cross-origin requests enabled for frontend
- **Testing**: Full test coverage for API and model components

## Troubleshooting

### Model not loading
- Verify pickle files exist in `artifacts/` directory
- Check file paths in `backend/config.py`
- Ensure pickle files are compatible with installed sklearn version

### API Connection Error
- Ensure backend is running on correct host/port
- Check `API_HOST` and `API_PORT` configuration
- Verify firewall settings

### Prediction Failures
- Check feature input dimensions match model expectations
- Verify features are valid numeric values
- Review model error logs for preprocessing issues

## Development

### Project Requirements
- Python 3.8+
- FastAPI for backend
- Streamlit for frontend
- scikit-learn for model
- pytest for testing

### Code Structure
- Clean separation between backend and frontend
- Modular model handler for easy integration
- Comprehensive type hints with Pydantic
- Proper error handling and logging

## License

Specify your license here.

## Contact

For questions or issues, please contact the development team.
