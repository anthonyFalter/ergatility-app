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