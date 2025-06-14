# src/rotab/ast/node.py
from pydantic import BaseModel
from typing import Optional, Any
from abc import ABC, abstractmethod
from pydantic import TypeAdapter


class Node(BaseModel, ABC):
    name: Optional[str] = None
    lineno: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def validate(self, context: Any) -> None:
        pass

    @abstractmethod
    def generate_script(self, context: Any = None) -> Any:
        pass

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "lineno": self.lineno,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name!r}, lineno={self.lineno!r})>"

    @classmethod
    def from_dict(cls, data: dict, schema_manager=None):
        return TypeAdapter(cls).validate_python(data, by_alias=True)
