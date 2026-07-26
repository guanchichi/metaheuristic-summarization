"""Task-profiled objective contracts."""

from src.objectives.factory import (
    aggregate_importance,
    build_objective_spec,
    validate_selector_for_task,
)

__all__ = [
    "aggregate_importance",
    "build_objective_spec",
    "validate_selector_for_task",
]
