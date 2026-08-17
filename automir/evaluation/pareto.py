from typing import Any, Dict, List, Tuple
import numpy as np


def dominates(
    candidate_a: Dict[str, float],
    candidate_b: Dict[str, float],
    objectives: List[Dict[str, str]],
) -> bool:
    """Determine if candidate_a Pareto-dominates candidate_b.

    Candidate A dominates B if:
    1. A is at least as good as B in all objectives
    2. A is strictly better than B in at least one objective
    """
    at_least_as_good = True
    strictly_better = False

    for obj in objectives:
        name = obj["name"]
        direction = obj.get("direction", "maximize").lower()
        val_a = float(candidate_a.get(name, -1e9 if direction == "maximize" else 1e9))
        val_b = float(candidate_b.get(name, -1e9 if direction == "maximize" else 1e9))

        if direction == "maximize":
            if val_a < val_b:
                at_least_as_good = False
                break
            if val_a > val_b:
                strictly_better = True
        else:  # minimize
            if val_a > val_b:
                at_least_as_good = False
                break
            if val_a < val_b:
                strictly_better = True

    return at_least_as_good and strictly_better


def fast_non_dominated_sort(
    candidates: List[Dict[str, Any]],
    objectives: List[Dict[str, str]],
) -> List[List[int]]:
    """Perform NSGA-II Fast Non-Dominated Sorting.

    Returns a list of fronts, where front[0] is the 1st Pareto front (non-dominated).
    """
    n = len(candidates)
    if n == 0:
        return []

    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(candidates[p], candidates[q], objectives):
                dominated_solutions[p].append(q)
                domination_count[q] += 1
            elif dominates(candidates[q], candidates[p], objectives):
                dominated_solutions[q].append(p)
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if len(next_front) > 0:
            fronts.append(next_front)
        i += 1

    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(
    front: List[int],
    candidates: List[Dict[str, Any]],
    objectives: List[Dict[str, str]],
) -> Dict[int, float]:
    """Calculate NSGA-II crowding distance for candidates in a given front."""
    l = len(front)
    if l == 0:
        return {}
    if l <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}

    for obj in objectives:
        name = obj["name"]
        # Sort front by objective value
        sorted_front = sorted(
            front, key=lambda idx: float(candidates[idx].get(name, 0.0))
        )

        # Boundary points receive infinite distance
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        min_val = float(candidates[sorted_front[0]].get(name, 0.0))
        max_val = float(candidates[sorted_front[-1]].get(name, 0.0))
        norm_range = max_val - min_val

        if norm_range > 1e-8:
            for k in range(1, l - 1):
                prev_val = float(candidates[sorted_front[k - 1]].get(name, 0.0))
                next_val = float(candidates[sorted_front[k + 1]].get(name, 0.0))
                distances[sorted_front[k]] += abs(next_val - prev_val) / norm_range

    return distances


def extract_pareto_front(
    candidates: List[Dict[str, Any]],
    objectives: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Extract the first Pareto front (non-dominated individuals) from a candidate list."""
    if not candidates:
        return []
    fronts = fast_non_dominated_sort(candidates, objectives)
    if not fronts:
        return []
    pareto_indices = fronts[0]
    return [candidates[i] for i in pareto_indices]


def compute_2d_hypervolume(
    points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Compute 2D Hypervolume for maximization objectives relative to a reference point."""
    if len(points) == 0:
        return 0.0

    # Filter points dominated by reference point
    valid = np.all(points >= reference_point, axis=1)
    pts = points[valid]
    if len(pts) == 0:
        return 0.0

    # Sort descending by x
    pts = pts[pts[:, 0].argsort()[::-1]]

    hv = 0.0
    current_y = reference_point[1]

    for p in pts:
        if p[1] > current_y:
            hv += (p[0] - reference_point[0]) * (p[1] - current_y)
            current_y = p[1]

    return float(hv)
