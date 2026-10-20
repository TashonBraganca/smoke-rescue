from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "brain"
EXCLUDED_ROBOT_FILES = {"evaluator.py", "viewer.py", "pipeline.py", "slam_compare.py", "compare_runs.py"}  # evaluators score against ground truth


def scan_strict_source(source: str, filename: str = "<source>") -> list[str]:
    """Return oracle-boundary violations for a robot-side Python source file."""
    violations: list[str] = []
    tree = ast.parse(source, filename=filename)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "brain.oracle" or alias.name.startswith("brain.oracle."):
                    violations.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports_oracle_module = module == "brain.oracle" or module.startswith(
                "brain.oracle."
            )
            imports_relative_oracle = node.level > 0 and (
                module == "oracle" or module.startswith("oracle.")
            )
            imports_oracle_from_brain = module == "brain" and any(
                alias.name == "oracle" for alias in node.names
            )
            if (
                imports_oracle_module
                or imports_relative_oracle
                or imports_oracle_from_brain
            ):
                violations.append(f"line {node.lineno}: imports oracle module")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "oracle" in node.value.casefold():
                violations.append(f"line {node.lineno}: contains oracle string literal")

    return violations


def scan_pipeline_source(source: str, filename: str = "<source>") -> list[str]:
    """Reject direct oracle paths while allowing the pipeline to import the evaluator."""
    violations: list[str] = []
    tree = ast.parse(source, filename=filename)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized = node.value.casefold().replace("\\", "/")
            path_parts = [part for part in normalized.split("/") if part]
            if "oracle" in path_parts:
                violations.append(f"line {node.lineno}: contains oracle path segment")

    return violations


def robot_side_modules() -> list[Path]:
    modules: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] == "oracle":
            continue
        if len(relative.parts) == 1 and relative.name in EXCLUDED_ROBOT_FILES:
            continue
        modules.append(path)
    return sorted(modules)


def test_robot_side_modules_do_not_reference_oracle() -> None:
    modules = robot_side_modules()
    assert modules, f"no robot-side modules found under {PACKAGE_ROOT}"

    violations: list[str] = []
    for path in modules:
        for violation in scan_strict_source(path.read_text(), str(path)):
            violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {violation}")

    assert not violations, "\n".join(violations)


def test_pipeline_does_not_open_oracle_paths() -> None:
    pipeline = PACKAGE_ROOT / "pipeline.py"
    if not pipeline.exists():
        return

    violations = scan_pipeline_source(pipeline.read_text(), str(pipeline))
    assert not violations, "\n".join(violations)


def test_scanner_flags_forbidden_oracle_import() -> None:
    source = "from brain.oracle import load_gt\n"
    assert scan_strict_source(source)


def test_scanner_flags_forbidden_oracle_path_literal() -> None:
    source = 'open(p / "oracle" / "x.txt")\n'
    assert scan_strict_source(source)


def test_scanner_accepts_clean_robot_source() -> None:
    source = 'from pathlib import Path\nPath("rgb") / "frame.png"\n'
    assert scan_strict_source(source) == []


def test_pipeline_scanner_allows_evaluator_import() -> None:
    source = "from brain import evaluator\n"
    assert scan_pipeline_source(source) == []
