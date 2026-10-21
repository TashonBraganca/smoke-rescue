from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "brain"
EXCLUDED_ROBOT_FILES = {"evaluator.py", "viewer.py", "pipeline.py", "slam_compare.py", "compare_runs.py", "nav_eval.py", "warm_eval.py", "rgb_people_eval.py"}  # evaluators score against ground truth


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
        # Offline camera evaluation lives here but the agent server never imports it.
        if relative == Path("agent/eval_vision.py"):
            continue
        modules.append(path)
    return sorted(modules)


def scan_import_graph(root: Path, starts: list[Path] | None = None) -> list[str]:
    """Follow local imports, including relative imports and package initialisers."""
    modules = {}
    for path in root.rglob("*.py"):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join([root.name, *parts])] = path
    names = {path: name for name, path in modules.items()}
    if starts is None:
        starts = [path for name, path in modules.items()
                  if name.startswith(f"{root.name}.agent.") and name != f"{root.name}.agent.eval_vision"]
    edges = {}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        imports = set()
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                imports.add(base)
                imports.update(f"{base}.{alias.name}" for alias in node.names)
        # Importing a submodule also runs each parent package's __init__.py.
        dependencies = set()
        for imported in imports | {name}:
            parts = imported.split(".")
            for end in range(1, len(parts) + 1):
                candidate = ".".join(parts[:end])
                if candidate in modules and candidate != name:
                    dependencies.add(candidate)
        edges[name] = sorted(dependencies)
    violations = []
    for start in starts:
        pending = [(names[start], [])]
        visited = set()
        while pending:
            name, chain = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            path = modules[name]
            chain = [*chain, str(path.relative_to(root))]
            direct = scan_strict_source(path.read_text(), str(path))
            if name == f"{root.name}.oracle" or name.startswith(f"{root.name}.oracle."):
                direct.append("reaches oracle module")
            violations.extend(f"{' -> '.join(chain)}: {violation}" for violation in direct)
            pending.extend((dependency, chain) for dependency in edges[name])
    return violations


def test_robot_side_modules_do_not_reference_oracle() -> None:
    modules = robot_side_modules()
    assert modules, f"no robot-side modules found under {PACKAGE_ROOT}"

    violations = scan_import_graph(PACKAGE_ROOT, modules)
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


@pytest.mark.parametrize('robot_import', [
    'import brain.evaluator',
    'from brain import evaluator',
    'from ..evaluator import load_gt',
    'from brain.bridge import helper',
    'import brain.bridge.child',
    'from brain.agent import eval_vision',
])
def test_import_graph_flags_evaluator_bridge(tmp_path, robot_import):
    package = tmp_path / 'brain'
    (package / 'agent').mkdir(parents=True)
    (package / 'oracle').mkdir()
    (package / 'bridge').mkdir()
    (package / '__init__.py').write_text('')
    (package / 'oracle/__init__.py').write_text('def load_gt(): pass\n')
    (package / 'evaluator.py').write_text('from brain.oracle import load_gt\n')
    (package / 'bridge/__init__.py').write_text('from brain.evaluator import load_gt as helper\n')
    (package / 'bridge/child.py').write_text('')
    (package / 'agent/__init__.py').write_text('')
    (package / 'agent/server.py').write_text(robot_import + '\n')
    (package / 'agent/eval_vision.py').write_text('from brain.evaluator import load_gt\n')
    violations = scan_import_graph(package, [package / 'agent/server.py'])
    assert violations, 'robot -> evaluator -> oracle must be rejected'
    assert any('agent/server.py' in v and 'evaluator.py' in v for v in violations)


def test_import_graph_checks_agent_helpers_not_imported_by_server(tmp_path):
    package = tmp_path / 'brain'
    (package / 'agent').mkdir(parents=True)
    (package / 'agent/server.py').write_text('')
    (package / 'agent/future_tool.py').write_text('from brain.evaluator import load_gt\n')
    (package / 'evaluator.py').write_text('from brain.oracle import load_gt\n')
    assert scan_import_graph(package, [p for p in (package / 'agent').rglob('*.py') if p.name != 'eval_vision.py'])


def test_import_graph_checks_modules_outside_agent(tmp_path):
    package = tmp_path / 'brain'
    (package / 'agent').mkdir(parents=True)
    (package / 'agent/server.py').write_text('from brain.io import reader\n')
    (package / 'io.py').write_text('from brain.evaluator import load_gt as reader\n')
    (package / 'evaluator.py').write_text('from brain.oracle import load_gt\n')
    assert scan_import_graph(package, [p for p in (package / 'agent').rglob('*.py') if p.name != 'eval_vision.py'])


def test_import_graph_allows_clean_cycle_and_unused_evaluator(tmp_path):
    package = tmp_path / 'brain'
    (package / 'agent').mkdir(parents=True)
    (package / 'agent/server.py').write_text('from brain import io\n')
    (package / 'io.py').write_text('from brain.agent import server\n')
    (package / 'evaluator.py').write_text('from brain.oracle import load_gt\n')
    assert scan_import_graph(package, [p for p in (package / 'agent').rglob('*.py') if p.name != 'eval_vision.py']) == []


@pytest.mark.parametrize("statement", [
    "import brain.evaluator",
    "from brain import evaluator",
    "from ..evaluator import load_survivors",
    "from .. import evaluator",
    "from brain.bridge import truth",
])
def test_robot_import_through_evaluator_is_rejected(tmp_path, monkeypatch, statement):
    package = tmp_path / "brain"
    (package / "agent").mkdir(parents=True)
    (package / "oracle").mkdir()
    (package / "__init__.py").write_text("")
    (package / "agent/__init__.py").write_text("")
    (package / "agent/robot.py").write_text(statement + "\n")
    (package / "bridge.py").write_text("from brain.evaluator import load_survivors as truth\n")
    (package / "evaluator.py").write_text("from brain.oracle import load_survivors\n")
    (package / "oracle/__init__.py").write_text("def load_survivors(): return []\n")
    monkeypatch.setattr(__import__(__name__, fromlist=['PACKAGE_ROOT']), "PACKAGE_ROOT", package)
    with pytest.raises(AssertionError, match=r"agent/robot.py.*evaluator.*oracle"):
        test_robot_side_modules_do_not_reference_oracle()
