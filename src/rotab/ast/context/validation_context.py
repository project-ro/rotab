from typing import Dict, Callable, Set, Optional
from pydantic import BaseModel, Field


class VariableInfo(BaseModel):
    type: str  # e.g., "dataframe"
    columns: Dict[str, str]


class ValidationContext(BaseModel):
    derive_func_path: Optional[str] = None
    transform_func_path: Optional[str] = None
    available_vars: Set[str]
    eval_scope: Dict[str, Callable]
    schemas: Dict[str, VariableInfo]
