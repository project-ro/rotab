from typing import List, Tuple, Optional, Dict, Type
from rotab.ast.template_node import TemplateNode
from rotab.ast.process_node import ProcessNode
from rotab.ast.step_node import StepNode
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.node import Node


class DagGenerator:
    def __init__(self, templates: List[TemplateNode]):
        self.templates = templates

    def build_step_edges(self, nodes: List[Node]) -> List[Tuple[Node, Node]]:
        edges: List[Tuple[Node, Node]] = []

        for idx, node in enumerate(nodes):
            for inp in node.get_inputs():
                for prev_node in reversed(nodes[:idx]):
                    if inp in prev_node.get_outputs():
                        edges.append((prev_node, node))
                        break
        return edges

    def build_template_edges(self) -> List[Tuple[TemplateNode, TemplateNode]]:
        name_to_template = {tpl.name: tpl for tpl in self.templates}
        edges = []

        for tpl in self.templates:
            for dep_name in getattr(tpl, "depends", []):
                dep_tpl = name_to_template.get(dep_name)
                if dep_tpl:
                    edges.append((dep_tpl, tpl))

        return edges

    def get_nodes(
        self,
        template_name: Optional[str] = None,
        process_name: Optional[str] = None,
        step_name: Optional[str] = None,
    ) -> List[Node]:
        result = []

        for tpl in self.templates:
            if template_name and tpl.name != template_name:
                continue
            result.append(tpl)

            for proc in tpl.get_children():
                if process_name and proc.name != process_name:
                    continue
                result.append(proc)

                for step in proc.get_children():
                    if step_name and step.name != step_name:
                        continue
                    result.append(step)
        return result

    def get_edges(
        self,
        template_name: Optional[str] = None,
        process_name: Optional[str] = None,
        step_name: Optional[str] = None,
    ) -> List[Tuple[Node, Node]]:
        nodes = self.get_nodes(template_name, process_name, step_name)
        node_set = set(nodes)
        edges = self.build_step_edges(nodes)
        return [e for e in edges if e[0] in node_set and e[1] in node_set]

    def generate_mermaid(self) -> str:
        lines = ["graph TB"]
        lines.append("%% Nodes")

        for tpl in self.templates:
            lines.append(f"%% Template: {tpl.name}")
            lines.append(f'subgraph T_{tpl.name} ["{tpl.name}"]')

            for proc in tpl.get_children():
                lines.append(f"  %% Process: {proc.name}")
                lines.append(f'  subgraph P_{proc.name} ["{proc.name}"]')

                proc_nodes = self.get_nodes(template_name=tpl.name, process_name=proc.name)
                proc_edges = self.get_edges(template_name=tpl.name, process_name=proc.name)

                for node in proc_nodes:
                    scoped_name = f"{tpl.name}__{node.name}"
                    if isinstance(node, InputNode):
                        lines.append(f'    I_{scoped_name}(["[I]{node.name}"])')
                    elif isinstance(node, OutputNode):
                        lines.append(f'    O_{scoped_name}(["[O]{node.name}"])')
                    elif isinstance(node, StepNode):
                        lines.append(f'    S_{scoped_name}(["[S]{node.name}"])')

                for src, dst in proc_edges:
                    lines.append(
                        f"    {self._node_prefix(src)}_{tpl.name}__{src.name} --> {self._node_prefix(dst)}_{tpl.name}__{dst.name}"
                    )

                lines.append("  end")  # end process
            lines.append("end")  # end template

        lines.append("%% Template Dependencies")
        for src_tpl, dst_tpl in self.build_template_edges():
            lines.append(f"T_{src_tpl.name} --> T_{dst_tpl.name}")

        return "\n".join(lines)

    def _node_prefix(self, node: Node) -> str:
        if isinstance(node, InputNode):
            return "I"
        elif isinstance(node, OutputNode):
            return "O"
        elif isinstance(node, StepNode):
            return "S"
        elif isinstance(node, ProcessNode):
            return "P"
        elif isinstance(node, TemplateNode):
            return "T"
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")
