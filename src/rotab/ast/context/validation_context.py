from typing import Dict, Callable, Set
from pydantic import BaseModel, Field


class VariableInfo(BaseModel):
    type: str  # e.g., "dataframe"
    columns: Dict[str, str]


class ValidationContext(BaseModel):
    available_vars: Set[str]
    eval_scope: Dict[str, Callable]
    schemas: Dict[str, VariableInfo]
