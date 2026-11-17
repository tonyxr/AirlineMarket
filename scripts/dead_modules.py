"""W4 / manual v3 rule 5: list package modules outside the paper import graph."""

import ast
import subprocess
from pathlib import Path

ROOTS = {"experiments", "campaign", "solvers", "eval", "learn", "game", "pomdp", "agents", "cli", "bench", "policy"}
paths = {".".join(p.with_suffix("").parts).removesuffix(".__init__"): p for p in Path("airline_market").rglob("*.py")}
graph = {name: set() for name in paths}
for name, path in paths.items():
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")[:len(package.split(".")) - node.level + 1] if node.level else []
            target = ".".join(base + ([node.module] if node.module else []))
            targets = [target] + [target + "." + alias.name for alias in node.names]
        for target in targets:
            parts = target.split(".")
            graph[name].update(".".join(parts[:k]) for k in range(1, len(parts) + 1) if ".".join(parts[:k]) in paths)
reachable = {name for name in paths if len(name.split(".")) > 1 and name.split(".")[1] in ROOTS}
pending = list(reachable)
while pending:
    for target in graph[pending.pop()] - reachable:
        reachable.add(target)
        pending.append(target)
print("Module\tFirst commit\tSource lines")
for name in sorted(paths.keys() - reachable):
    path = paths[name]
    dates = subprocess.check_output(["git", "log", "--diff-filter=A", "--format=%aI", "--reverse", "--", str(path)], text=True).splitlines()
    lines = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in path.read_text().splitlines())
    print(f"{path}\t{dates[0] if dates else 'uncommitted'}\t{lines}")
