from automir.automl.candidate import CandidateConfig
from automir.automl.evolutionary import EvolutionaryParetoSearch
from automir.automl.search_space import SearchSpace


def test_search_space_sample():
    cand = SearchSpace.sample_candidate(generation=0)
    assert isinstance(cand, CandidateConfig)
    assert cand.validate()


def test_crossover_validity():
    parent_a = SearchSpace.sample_candidate(generation=0)
    parent_b = SearchSpace.sample_candidate(generation=0)
    parent_a.representation = "logmel"
    parent_b.representation = "tempogram"

    child = SearchSpace.crossover(parent_a, parent_b, generation=1)
    assert isinstance(child, CandidateConfig)
    assert child.generation == 1
    assert child.representation in ["logmel", "tempogram"]
    assert child.validate()


def test_mutation_validity():
    cand = SearchSpace.sample_candidate(generation=0)
    mutant = SearchSpace.mutate(cand, mutation_prob=0.8)

    assert isinstance(mutant, CandidateConfig)
    assert mutant.validate()


def test_evolutionary_rank_assignment():
    searcher = EvolutionaryParetoSearch(
        objectives=[
            {"name": "tempo_acc_4", "direction": "maximize"},
            {"name": "latency_ms", "direction": "minimize"},
        ]
    )

    pop = [
        CandidateConfig(candidate_id="c1", metrics={"tempo_acc_4": 90.0, "latency_ms": 10.0}),
        CandidateConfig(candidate_id="c2", metrics={"tempo_acc_4": 80.0, "latency_ms": 20.0}),
    ]

    ranked_pop = searcher._assign_ranks_and_crowding(pop)
    assert ranked_pop[0].rank == 0  # c1 dominates c2
    assert ranked_pop[1].rank == 1
