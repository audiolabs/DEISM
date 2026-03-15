import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Set


TARGET_METHODS = [
    "init_params",
    "update_room",
    "update_wall_materials",
    "update_freqs",
    "update_source_receiver",
    "update_directivities",
    "run_DEISM",
]


class ParamsAccessVisitor(ast.NodeVisitor):
    def __init__(self):
        self.reads: Set[str] = set()
        self.writes: Set[str] = set()

    @staticmethod
    def _is_self_params(node: ast.AST) -> bool:
        # self.params["x"]
        if not isinstance(node, ast.Subscript):
            return False
        value = node.value
        if not isinstance(value, ast.Attribute):
            return False
        return (
            isinstance(value.value, ast.Name)
            and value.value.id == "self"
            and value.attr == "params"
        )

    @staticmethod
    def _extract_key(node: ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
        return None

    def visit_Subscript(self, node: ast.Subscript):
        if self._is_self_params(node):
            key = self._extract_key(node)
            if key is not None:
                if isinstance(node.ctx, ast.Store):
                    self.writes.add(key)
                else:
                    self.reads.add(key)
        self.generic_visit(node)


def analyze_deism_class(source_path: Path) -> Dict[str, Dict[str, List[str]]]:
    src = source_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    result: Dict[str, Dict[str, List[str]]] = {}

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DEISM":
            class_node = node
            break
    if class_node is None:
        raise RuntimeError("DEISM class not found.")

    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_METHODS:
            visitor = ParamsAccessVisitor()
            visitor.visit(node)
            result[node.name] = {
                "reads": sorted(visitor.reads),
                "writes": sorted(visitor.writes),
            }
    return result


def write_markdown(dep_map: Dict[str, Dict[str, List[str]]], md_path: Path):
    lines = [
        "# DEISM Parameter Dependency Map",
        "",
        "Auto-generated from static analysis of `deism/core_deism.py`.",
        "",
    ]
    for method in TARGET_METHODS:
        if method not in dep_map:
            continue
        reads = dep_map[method]["reads"]
        writes = dep_map[method]["writes"]
        lines.append(f"## `{method}`")
        lines.append("")
        lines.append(
            f"- Reads ({len(reads)}): " + (", ".join(f"`{k}`" for k in reads) or "None")
        )
        lines.append(
            f"- Writes ({len(writes)}): "
            + (", ".join(f"`{k}`" for k in writes) or "None")
        )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DEISM param read/write dependencies."
    )
    parser.add_argument(
        "--source",
        default="deism/core_deism.py",
        help="Path to core_deism.py",
    )
    parser.add_argument(
        "--out-json",
        default="tools/reports/deism_dependency_map.json",
    )
    parser.add_argument(
        "--out-md",
        default="tools/reports/deism_dependency_map.md",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    dep_map = analyze_deism_class(source_path)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dep_map, indent=2), encoding="utf-8")
    write_markdown(dep_map, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
