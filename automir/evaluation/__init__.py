"""Evaluation metrics, efficiency benchmarking, and Pareto front analysis."""

from automir.evaluation.metrics import (
    compute_tempo_metrics,
    compute_classification_metrics,
)
from automir.evaluation.efficiency import benchmark_efficiency
from automir.evaluation.pareto import (
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    extract_pareto_front,
    compute_2d_hypervolume,
)

__all__ = [
    "compute_tempo_metrics",
    "compute_classification_metrics",
    "benchmark_efficiency",
    "dominates",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "extract_pareto_front",
    "compute_2d_hypervolume",
]
