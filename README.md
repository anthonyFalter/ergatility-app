# Ergatility Employee Retention Prediction System

Ergatility is a unified simulation platform and RESTful API built to model, simulate, and analyze employee turnover dynamics. By combining predictive analytics with programmatic access, Ergatility enables organizations and developers to evaluate retention scenarios, assess attrition risks, and integrate workforce simulation data directly into enterprise HR workflows.

- **Streamlit Interactive Dashboard:** [https://ergatility.streamlit.app/](https://ergatility.streamlit.app/)

- **Render Backend API:** [https://ergatility-backend-api.onrender.com](https://ergatility-backend-api.onrender.com)


**Features:**
- Turnover Simulation Engine: Runs predictive models to simulate workforce attrition across various organizational scenarios and parameters.

- RESTful API Service: Exposes clean, high-performance endpoints built with FastAPI to programmatically execute simulations, query prediction models, and check engine health.

- Integrated Environment: Package both analytics models and API infrastructure within a single, modular codebase for streamlined deployment and testing.


*Note: The API is hosted using Render, an operational delay of up to 1 minute is to be expected so please wait while your payload gets processed.*

## Project Structure

```
ergatility-app/
├── artifacts/                     
│   └── rf_cv_model.pickle          # Trained Random Forest model
│
├── backend/                        
│   ├── main.py                     # FastAPI application & routes
│   ├── schemas.py                  # Pydantic data validation schemas
│   ├── model_handler.py            # Model loading & inference logic
│   ├── config.py                   # Configuration & environment paths
│   └── __init__.py
│
├── frontend/                       
│   ├── app.py                      # Streamlit interactive UI
│   ├── utils.py                    # API client & request helpers
│   └── __init__.py
│
├── tests/                          
│   ├── test_api.py                 # FastAPI endpoint unit tests
│   └── test_model_handler.py       # Model handler unit tests
│
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

## Installation

### Prerequisites
- Python 3.12+
- pip (Python package manager)

### Setup

1. Clone or navigate into the project directory:
```bash
cd ergatility-app
```

2. Create and activate a virtual environment:

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install project dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Backend API

Start the FastAPI server:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

- Interactive API Docs (Swagger UI): `http://localhost:8000/docs`
- Alternative Docs (ReDoc): `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

### Running the Frontend

In a separate terminal, activate the virtual environment and start the Streamlit application:

```bash
streamlit run frontend/app.py
```

The application interface will launch at `http://localhost:8501`.

## API Endpoints

### Health Check
- **GET** `/health`
  - Returns backend API service status and model loading state.
  - Response: `{"status": "healthy", "model_loaded": true, "version": "1.0.0"}`

### Predict Employee Churn
- **POST** `/predict`
  - Input JSON:
    ```json
    {
      "satisfaction_level": 0.75,
      "last_evaluation": 0.78,
      "number_project": 4,
      "average_montly_hours": 200,
      "time_spend_company": 3,
      "work_accident": 0,
      "promotion_last_5years": 0,
      "salary": "medium",
      "department": "sales"
    }
    ```
  - Response JSON:
    ```json
    {
      "prediction": 0,
      "probability": 0.084,
      "risk_level": "Low Risk"
    }
    ```

## Example API Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "satisfaction_level": 0.09,
    "last_evaluation": 0.88,
    "number_project": 6,
    "average_montly_hours": 275,
    "time_spend_company": 4,
    "work_accident": 0,
    "promotion_last_5years": 0,
    "salary": "low",
    "department": "sales"
  }'
```

### Using Python

```python
import requests

payload = {
    "satisfaction_level": 0.75,
    "last_evaluation": 0.78,
    "number_project": 4,
    "average_montly_hours": 200,
    "time_spend_company": 3,
    "work_accident": 0,
    "promotion_last_5years": 0,
    "salary": "medium",
    "department": "sales"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

## Testing

Run the automated test suite using `pytest`:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_api.py
pytest tests/test_model_handler.py

# Run with verbose output
pytest -v

# Run with coverage report
pip install pytest-cov
pytest --cov=backend --cov=frontend
```

## Configuration & Environment Variables

Key settings can be configured via environment variables or modified in `backend/config.py` and `frontend/utils.py`:

- **`BACKEND_URL`**: Base URL used by the Streamlit frontend to communicate with the FastAPI server (Default: `http://localhost:8000`).
- **`MODEL_PATH`**: File system location of the trained model binary (Default: `artifacts/rf_cv_model.pickle`).

## Troubleshooting

### API Service Offline in Streamlit
- Ensure the FastAPI server is running on `http://localhost:8000` before submitting prediction forms in Streamlit.
- Check that `BACKEND_URL` is configured correctly if deployed across external servers (e.g., Railway or Render).

### ModuleNotFoundError: No module named 'frontend' or 'backend'
- Run Python commands with module execution context: `python -m uvicorn backend.main:app --reload`.
- Ensure you launch Streamlit using `streamlit run frontend/app.py` from the root project folder.

### Validation Errors (422 Unprocessable Entity)
- Ensure all numeric fields adhere to defined constraints (e.g., `satisfaction_level` between `0.0` and `1.0`).
- Ensure categorical fields match expected values (`salary`: `"low"`, `"medium"`, or `"high"`).
