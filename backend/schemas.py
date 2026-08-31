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
    
    time_spend_company: int = Field(
        ...,
        ge=2,
        le=10,
        description="Years spent at the company (range: 2–10)",
        examples=[3],
    )
    work_accident: Literal[0, 1] = Field(
        ...,
        description="0 = No work accident, 1 = Work accident occurred",
        examples=[0],
    )
    promotion_last_5years: Literal[0, 1] = Field(
        ...,
        description="0 = Not promoted in last 5 years, 1 = Promoted",
        examples=[0],
    )
    salary: Literal["low", "medium", "high"] = Field(
        ...,
        description="Salary level tier ('low', 'medium', 'high')",
        examples=["medium"],
    )
    department: Literal[
        "sales",
        "accounting",
        "hr",
        "technical",
        "support",
        "management",
        "IT",
        "product_mng",
        "marketing",
        "RandD",
    ] = Field(
        ...,
        description="Department name",
        examples=["sales"],
    )
    
    class ChurnPredictionOutput(BaseModel):
        prediction: int = Field(
            ...,
            description='Predicted class: 1 (Leave) or 0 (Stay)'
        )
        
        probability: float = Field(
            ...,
            description='Probability of churn (range: 0.0-1.0)'
        )
        
        risk_level: str = Field(
            ...,
            description="Risk evaluation label ('High Risk' or 'Low Risk')"
        )