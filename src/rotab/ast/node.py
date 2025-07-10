import os
from pydantic import BaseModel
from typing import Optional, Any, List
from abc import ABC, abstractmethod

try:
    from pydantic import TypeAdapter

    _USE_TYPE_ADAPTER = True
except ImportError:
    _USE_TYPE_ADAPTER = False


class Node(BaseModel, ABC):
    name: Optional[str] = None
    lineno: Optional[int] = None

    def __hash__(self):
        return hash((type(self), self.name))

    def __eq__(self, other):
        return isinstance(other, Node) and type(self) == type(other) and self.name == other.name

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def validate(self, context: Any) -> None:
        pass

    @abstractmethod
    def generate_script(self, backend: str = "pandas", context: Any = None) -> Any:
        pass

    def get_children(self) -> List["Node"]:
        return []

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "lineno": self.lineno,
        }

    def get_dag_node_name(self) -> str:
        prefix_map = {"TemplateNode": "T", "ProcessNode": "P", "StepNode": "S", "InputNode": "I", "OutputNode": "O"}
        cls_name = self.__class__.__name__
        prefix = prefix_map.get(cls_name, "NotSupported")
        if hasattr(self, "path") and self.path:
            filename = os.path.basename(self.path)
            return f"[{prefix}]{filename}"
        elif self.name:
            return f"[{prefix}]{self.name}"
        else:
            return f"[{prefix}]{id(self)}"

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name!r}, lineno={self.lineno!r})>"

    @classmethod
    def from_dict(cls, data: dict, schema_manager=None):
        if _USE_TYPE_ADAPTER:
            return TypeAdapter(cls).validate_python(data)
        else:
            return cls(**data)  # Pydantic v1 fallback
