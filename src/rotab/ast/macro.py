from rotab.ast.step import StepNode
from rotab.ast.context.validation_context import Scope
from typing import Dict, Any, Optional, List


class UseMacro(StepNode):
    def __init__(self, name: str, macro_name: str, args: Dict[str, Any], lineno: Optional[int] = None):
        super().__init__(name, [], None, lineno)
        self.macro_name = macro_name
        self.args = args
        self.expanded_steps: List[StepNode] = []

    def validate(self, scope: Scope) -> None:
        if self.macro_name not in scope["macros"]:
            raise ValueError(f"[{self.name}] Macro `{self.macro_name}` not defined.")
        macro_def = scope["macros"][self.macro_name]
        # 仮：単純にコピーして引数未展開（引数展開が必要なら別ロジック）
        self.expanded_steps = macro_def.steps

        for step in self.expanded_steps:
            step.validate(scope)

    def generate_script(self) -> List[str]:
        lines: List[str] = []
        for step in self.expanded_steps:
            lines.extend(step.generate_script())
        return lines
