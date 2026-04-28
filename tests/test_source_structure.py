from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _collect_python_files():
    package_root = Path(__file__).resolve().parents[1] / "bayRing"
    for path in sorted(package_root.glob("*.py")):
        yield path


def _collect_functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            functions.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            functions.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return functions


@pytest.mark.parametrize("module_path", list(_collect_python_files()))
def test_module_has_parsable_functions(module_path):
    functions = _collect_functions(module_path)
    if module_path.name == "__init__.py":
        assert not functions
    else:
        assert functions, f"Expected to find functions in {module_path.name}"


@pytest.mark.parametrize(
    "module_path, function_node",
    [
        pytest.param(path, func, id=f"{path.name}::{func.name}")
        for path in _collect_python_files()
        for func in _collect_functions(path)
    ],
)
def test_all_functions_have_body(module_path: Path, function_node: ast.FunctionDef):
    assert len(function_node.body) > 0, f"Function {function_node.name} in {module_path.name} is empty"
