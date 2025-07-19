import os
from typing import List, Dict, Optional
from rotab.ast.template_node import TemplateNode
from rotab.ast.context.validation_context import ValidationContext
from rotab.runtime.dag_generator import DagGenerator

BACKEND = "pandas"  # or "polars", depending on your use case


class CodeGenerator:
    def __init__(self, templates: List[TemplateNode], backend: str, context: ValidationContext):
        self.templates = templates
        self.backend = backend
        self.context = context
        self.dag = DagGenerator(templates)

    def _resolve_template_order(self) -> List[TemplateNode]:
        from collections import defaultdict, deque

        edges = self.dag.build_template_edges()
        name_to_tpl = {tpl.name: tpl for tpl in self.templates}

        graph = defaultdict(list)
        indegree = defaultdict(int)

        for src, dst in edges:
            graph[src.name].append(dst.name)
            indegree[dst.name] += 1

        for name in name_to_tpl:
            indegree.setdefault(name, 0)

        queue = deque([name for name in name_to_tpl if indegree[name] == 0])
        sorted_names = []

        while queue:
            name = queue.popleft()
            sorted_names.append(name)
            for nbr in graph[name]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)

        if len(sorted_names) != len(name_to_tpl):
            raise ValueError("Cyclic dependency detected among templates.")

        return [name_to_tpl[name] for name in sorted_names]

    def generate(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns:
            {
                "template_name": {
                    "process_name": [line1, line2, ...],
                    ...
                },
                ...
            }
        """
        result = {}
        for template in self._resolve_template_order():
            result[template.name] = template.generate_script(self.backend, self.context)
        return result

    def write_all(self, source_dir: str, selected_processes: Optional[List[str]] = None) -> None:
        """
        Write scripts to source_dir/template_name/process_name.py
        Also generates source_dir/main.py that calls all (or selected) processes in dependency order.
        """
        all_calls = []

        for template in self._resolve_template_order():
            template_dir = os.path.join(source_dir, template.name)
            os.makedirs(template_dir, exist_ok=True)

            script_map = template.generate_script(self.backend, self.context)

            for process_name, lines in script_map.items():
                path = os.path.join(template_dir, f"{process_name}.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                    f.write("\n")
                all_calls.append((process_name, template.name))

        # main.py を生成
        main_path = os.path.join(source_dir, "main.py")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write("import os\nimport sys\n")
            f.write("project_root = os.path.dirname(os.path.abspath(__file__))\n")
            f.write("sys.path.insert(0, project_root)\n\n")

            for proc_name, tpl_name in all_calls:
                if selected_processes is None or proc_name in selected_processes:
                    f.write(f"from {tpl_name}.{proc_name} import {proc_name}\n")

            f.write("\n\nif __name__ == '__main__':\n")
            for proc_name, _ in all_calls:
                if selected_processes is None or proc_name in selected_processes:
                    f.write(f"    {proc_name}()\n")
