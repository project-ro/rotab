from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class IOperation(ABC):
    @abstractmethod
    def execute(self, inputs: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.DataFrame:
        pass
