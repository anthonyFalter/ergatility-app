from typing import Literal
from pydantic import BaseModel, Field

class ChurnInputSchema(BaseModel):
    satisfaction_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Employee satisfaction score between 0.0 and 1.0',
        example=[0.75],
    )
    
    last_evaluation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Last performance evaluation score between 0.0 and 1.0',
        examples=[0.78]
    )
    
    number_project: int = Field(
        ...,
        ge=2,
        le=7,
        description='Number of assigned projects (range:2-7)',
        examples=[4]
    )
    
    average_montly_hours: int = Field(
        ...,
        ge=90,
        le=310,
        description='Average monthly hours worked (range: 90-310)',
        examples=[200],
    )