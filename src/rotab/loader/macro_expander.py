from copy import deepcopy
from typing import Any, Dict, List
import re
from rotab.utils.logger import get_logger

logger = get_logger()


class MacroExpander:
    def __init__(self, macro_definitions: Dict[str, Dict[str, Any]]):
        self.macro_map: Dict[str, List[Dict[str, Any]]] = {
            name: macro["steps"] for name, macro in macro_definitions.items()
        }

    def expand(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded_steps = []
        for step in steps:
            if "use" in step:
                expanded_steps.extend(self._expand_macro_step(step))
            else:
                expanded_steps.append(step)
        return expanded_steps

    def _expand_macro_step(self, use_step: Dict[str, Any]) -> List[Dict[str, Any]]:
        macro_name = use_step["use"]
        if macro_name not in self.macro_map:
            raise KeyError(f"Macro `{macro_name}` not defined.")

        args = use_step.get("args", {})
        if not isinstance(args, dict):
            raise TypeError("`args` must be a dictionary.")

        caller_step_name = use_step["name"]

        raw_steps = self.macro_map[macro_name]
        if not isinstance(raw_steps, list):
            raise ValueError("Macro steps must be a list.")

        expanded_steps = []
        for raw_step in raw_steps:
            step = deepcopy(raw_step)
            self._replace_macro_vars(step, use_step)
            step["name"] = caller_step_name
            expanded_steps.append(step)

        logger.info(f"Expanded macro `{macro_name}` into {len(expanded_steps)} steps.")
        return expanded_steps

    def _replace_macro_vars(self, step: Dict[str, Any], use_step: Dict[str, Any]) -> None:
        caller_with = use_step.get("with")
        caller_as = use_step.get("as")
        args = use_step.get("args", {})

        if caller_with is None:
            raise ValueError(f"Macro call is missing required `with` field: {use_step}")
        if caller_as is None:
            raise ValueError(f"Macro call is missing required `as` field: {use_step}")

        def replace(val: Any) -> Any:
            if isinstance(val, str):
                val = val.replace("${caller.with}", str(caller_with))
                val = val.replace("${caller.as}", str(caller_as))

                for match in re.findall(r"\$\{args\.([^\}]+)\}", val):
                    if match not in args:
                        raise KeyError(f"Missing argument: args.{match}")
                    replacement = args[match]
                    if isinstance(replacement, (list, dict)):
                        if val.strip() == f"${{args.{match}}}":
                            return replacement
                        val = val.replace(f"${{args.{match}}}", str(replacement))
                    else:
                        val = val.replace(f"${{args.{match}}}", str(replacement))
                return val

            elif isinstance(val, list):
                return [replace(x) for x in val]
            elif isinstance(val, dict):
                return {k: replace(v) for k, v in val.items()}
            else:
                return val

        for k in list(step.keys()):
            step[k] = replace(step[k])
