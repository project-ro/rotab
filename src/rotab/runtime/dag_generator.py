from typing import List, Tuple, Optional, Dict, Type
from rotab.ast.template_node import TemplateNode
from rotab.ast.process_node import ProcessNode
from rotab.ast.step_node import StepNode
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
