"""
Orchestration — Dependency Graph Engine.

Manages workflow step dependencies:
  - DAG-based dependency resolution
  - Topological sort for execution order
  - Readiness checks (can a stage run?)
  - Cycle detection
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class DependencyGraph:
    """
    DAG-based dependency manager for pipeline stages.

    Provides topological sort, readiness checks, and cycle detection.
    """

    def __init__(self):
        cfg = _load_config().get("dependencies", {})
        self._deps: dict[str, list[str]] = defaultdict(list)
        self._completed: set[str] = set()

        # Load configured dependencies
        for stage, info in cfg.items():
            if isinstance(info, dict) and "requires" in info:
                self._deps[stage] = list(info["requires"])

    def add_dependency(self, stage: str, depends_on: str) -> None:
        """Add a dependency: stage depends on depends_on."""
        if depends_on not in self._deps[stage]:
            self._deps[stage].append(depends_on)

    def get_dependencies(self, stage: str) -> list[str]:
        """Get all direct dependencies for a stage."""
        return list(self._deps.get(stage, []))

    def mark_completed(self, stage: str) -> None:
        """Mark a stage as completed."""
        self._completed.add(stage)

    def is_ready(self, stage: str) -> bool:
        """Check if all dependencies for a stage are completed."""
        return all(dep in self._completed for dep in self._deps.get(stage, []))

    def unmet(self, stage: str) -> list[str]:
        """Return unmet dependencies for a stage."""
        return [dep for dep in self._deps.get(stage, []) if dep not in self._completed]

    def reset(self) -> None:
        """Reset completion state (keep dependency graph)."""
        self._completed.clear()

    def topological_sort(self) -> list[str]:
        """
        Return stages in valid execution order (Kahn's algorithm).
        Raises ValueError on cycles.
        """
        # Build in-degree map
        all_stages = set(self._deps.keys())
        for deps in self._deps.values():
            all_stages.update(deps)

        in_degree: dict[str, int] = {s: 0 for s in all_stages}
        adj: dict[str, list[str]] = defaultdict(list)

        for stage, deps in self._deps.items():
            for dep in deps:
                adj[dep].append(stage)
                in_degree[stage] = in_degree.get(stage, 0) + 1

        queue = deque(s for s in all_stages if in_degree[s] == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(all_stages):
            raise ValueError("Cycle detected in dependency graph")

        return result

    def has_cycle(self) -> bool:
        """Check if the dependency graph contains a cycle."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def get_runnable(self) -> list[str]:
        """Get all stages that are ready to run (deps met, not completed)."""
        all_stages = set(self._deps.keys())
        for deps in self._deps.values():
            all_stages.update(deps)
        return [
            s for s in all_stages
            if s not in self._completed and self.is_ready(s)
        ]

    def summary(self) -> dict:
        return {
            "stages": len(set(self._deps.keys()) | set(d for deps in self._deps.values() for d in deps)),
            "edges": sum(len(deps) for deps in self._deps.values()),
            "completed": len(self._completed),
            "runnable": len(self.get_runnable()),
        }
