from typing import List, Optional
from pydantic import BaseModel, Field
from rotab.ast.process_node import ProcessNode
from rotab.ast.context.validation_context import ValidationContext
from rotab.ast.node import Node


class TemplateNode(Node):
    name: str
    depends: List[str] = Field(default_factory=list)
    processes: List[ProcessNode] = Field(default_factory=list)
    lineno: Optional[int] = None

    def validate(self, context: ValidationContext) -> None:
        for process in self.processes:
            process.validate(context)

    def get_children(self) -> List[Node]:
        return self.processes

    def generate_script(self, backend: str, context: ValidationContext) -> dict:
        """
        Returns a dict mapping process name to its script lines.
        Example:
            {
                "filter_users": [...],
                "enrich_transactions": [...],
            }
        """
        script_map = {}

        for process in self.processes:
            script_map[process.name] = process.generate_script(backend, context)

        return script_map

    def to_dict(self) -> dict:
        return {
            "type": "TemplateNode",
            "name": self.name,
            "depends": self.depends,
            "processes": [p.to_dict() for p in self.processes],
        }
