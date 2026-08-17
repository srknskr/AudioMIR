import numpy as np
from automir.evaluation.pareto import (
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    extract_pareto_front,
    compute_2d_hypervolume,
)


def test_pareto_dominance():
    objs = [
        {"name": "acc", "direction": "maximize"},
        {"name": "latency", "direction": "minimize"},
    ]

    # Candidate A: Acc 90%, Latency 10ms
    # Candidate B: Acc 80%, Latency 20ms (A dominates B)
    cand_a = {"acc": 90.0, "latency": 10.0}
    cand_b = {"acc": 80.0, "latency": 20.0}
    cand_c = {"acc": 95.0, "latency": 25.0}  # Trade-off with A: higher acc, higher latency

    assert dominates(cand_a, cand_b, objs)
    assert not dominates(cand_b, cand_a, objs)
    assert not dominates(cand_a, cand_c, objs)
    assert not dominates(cand_c, cand_a, objs)


def test_fast_non_dominated_sort_and_front_extraction():
    objs = [
        {"name": "acc", "direction": "maximize"},
        {"name": "latency", "direction": "minimize"},
    ]

    candidates = [
        {"id": 0, "acc": 90.0, "latency": 10.0},  # Pareto Front 0
        {"id": 1, "acc": 95.0, "latency": 20.0},  # Pareto Front 0
        {"id": 2, "acc": 80.0, "latency": 30.0},  # Dominated by 0 and 1
    ]

    fronts = fast_non_dominated_sort(candidates, objs)
    assert len(fronts) >= 2
    assert set(fronts[0]) == {0, 1}
    assert 2 in fronts[1]

    pareto_front = extract_pareto_front(candidates, objs)
    assert len(pareto_front) == 2
    assert {c["id"] for c in pareto_front} == {0, 1}


def test_crowding_distance():
    objs = [
        {"name": "acc", "direction": "maximize"},
        {"name": "latency", "direction": "minimize"},
    ]

    candidates = [
        {"acc": 70.0, "latency": 5.0},
        {"acc": 80.0, "latency": 10.0},
        {"acc": 90.0, "latency": 15.0},
    ]

    front = [0, 1, 2]
    distances = calculate_crowding_distance(front, candidates, objs)
    assert distances[0] == float("inf")
    assert distances[2] == float("inf")
    assert distances[1] > 0.0


def test_hypervolume():
    pts = np.array([
        [10.0, 5.0],
        [5.0, 10.0],
    ])
    ref = np.array([0.0, 0.0])
    hv = compute_2d_hypervolume(pts, ref)
    # Area = 10*5 + 5*(10-5) = 50 + 25 = 75
    assert hv == 75.0
