import logging
from pathlib import Path
import pickle
import pandas as pd
from backend.schemas import ChurnInputSchema

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.load_model()
        
    def load_model(self) -> None:
        '''Load the trained machine learning model from the pickle file'''
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
            
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
                
            self.is_loaded = True
            logger.info(f"Succesfully loaded model from {self.model_path.name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            raise e
        
    def preprocess_input(self, data: ChurnInputSchema) -> pd.DataFrame:
        '''Transform Pydantic input schema to feature-aligned pandas Dataframe'''
        
        raw_dict = data.model_dump()
        
        # Map Ordinal Salary
        salary_map = {'low': 0, 'medium': 1, 'high': 2}
        salary_val = salary_map.get(raw_dict.pop("salary"),1)
        
        # Base Features dictionary
        processed_data = {
            'satisfaction_level': raw_dict['satisfaction_level'],
            'last_evaluation': raw_dict['last_evaluation'],
            'number_project': raw_dict['number_project'],
            'average_montly_hours': raw_dict['average_montly_hours'],
            'time_spend_company': raw_dict['time_spend_company'],
            'work_accident': raw_dict['work_accident'],
            'promotion_last_5years': raw_dict['promotion_last_5years'],
            'salary': salary_val,
                                             
        }
        
        # Apply One-Hot Encoding for `Departments`
        all_departments = [
            "RandD",
            "accounting",
            "hr",
            "management",
            "marketing",
            "product_mng",
            "sales",
            "support",
            "technical",
        ] # `IT` not included as it serves as implicit baseline
        
        selected_dept = raw_dict["department"]
        for dept in all_departments:
            processed_data[f"department_{dept}"] = (
                1 if selected_dept == dept else 0
            )
            
        df = pd.DataFrame([processed_data])
        
        # Column Order Alignment
        if hasattr(self.model, 'feature_names_in_'):
            df = df[self.model.feature_names_in_]
            
        return df

def predict(self, data: ChurnInputSchema) -> tuple[int, float, str]:
    '''Execute model prediction and return formatted result metrics'''
    
    if not self.is_loaded or self.model is None:
        raise RuntimeError('Model is not loaded.')
    
    input_df = self.preprocess_input(data)
    
    # Class prediction
    prediction = int(self.model.predict(input_df)[0])
    probabilities = self.mode.predict_proba(input_df)[0]
    probability = float(probabilities[1])
    
    # Risk evaluation label
    risk_level = 'High Risk' if prediction == 1 else 'Low Risk'
    
    return prediction, probability, risk_level