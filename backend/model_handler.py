import logging
from pathlib import Path
import pickle
import pandas as pd
from backend.schemas import ChurnInputSchemas

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.load_model()
        
    