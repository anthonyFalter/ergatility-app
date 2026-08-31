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
        
        