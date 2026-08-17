import copy
import random
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from automir.automl.base import BaseSearchStrategy
from automir.automl.candidate import CandidateConfig
from automir.automl.search_space import SearchSpace
from automir.evaluation.pareto import (
    fast_non_dominated_sort,
    calculate_crowding_distance,
)


class EvolutionaryParetoSearch(BaseSearchStrategy):
    """Native NSGA-II Multi-Objective Evolutionary Algorithm for Neural Architecture Search.

    Implements:
    - Non-dominated sorting
    - Crowding distance calculation
    - Binary tournament selection (Rank -> Crowding Distance)
    - Architecture & Hyperparameter Crossover & Mutation
    - (N + N) -> N Elitist survivor selection
    """

    def __init__(
        self,
        population_size: int = 10,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.35,
        tournament_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

    def _assign_ranks_and_crowding(
        self, population: List[CandidateConfig]
    ) -> List[CandidateConfig]:
        """Compute and annotate Pareto ranks and crowding distances on candidate objects."""
        if not population:
            return population

        metrics_list = [c.metrics for c in population]
        fronts = fast_non_dominated_sort(metrics_list, self.objectives)

        for rank, front in enumerate(fronts):
            crowding_distances = calculate_crowding_distance(
                front, metrics_list, self.objectives
            )
            for idx in front:
                population[idx].rank = rank
                population[idx].crowding_distance = crowding_distances.get(idx, 0.0)

        return population

    def _tournament_select(self, population: List[CandidateConfig]) -> CandidateConfig:
        """Binary tournament selection: prefer lower rank; break ties with higher crowding distance."""
        k = min(self.tournament_size, len(population))
        contestants = random.sample(population, k)
        # Sort contestants: rank ascending (0 is best), crowding_distance descending (inf is best)
        contestants.sort(key=lambda c: (c.rank, -c.crowding_distance))
        return contestants[0]

    def search(
        self,
        evaluations: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        callback: Optional[Callable[[CandidateConfig], None]] = None,
    ) -> List[CandidateConfig]:
        self.history = []
        eval_count = 0
        gen = 0

        pop_size = min(self.population_size, evaluations)

        # 1. Initialize random population P_0
        population: List[CandidateConfig] = []
        for _ in range(pop_size):
            if eval_count >= evaluations:
                break
            cand = SearchSpace.sample_candidate(generation=0)
            cand = self.evaluate_candidate(
                candidate=cand,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
            )
            population.append(cand)
            self.history.append(cand)
            eval_count += 1
            if callback is not None:
                callback(cand)

        population = self._assign_ranks_and_crowding(population)

        # 2. Main generation loop
        while eval_count < evaluations:
            gen += 1
            offspring_pop: List[CandidateConfig] = []

            while len(offspring_pop) < pop_size and eval_count < evaluations:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)

                if random.random() < self.crossover_prob:
                    child = SearchSpace.crossover(parent_a, parent_b, generation=gen)
                else:
                    child = copy.deepcopy(parent_a)
                    child.generation = gen

                child = SearchSpace.mutate(child, mutation_prob=self.mutation_prob)

                child = self.evaluate_candidate(
                    candidate=child,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    device=device,
                )
                offspring_pop.append(child)
                self.history.append(child)
                eval_count += 1
                if callback is not None:
                    callback(child)

            # 3. Elitist Selection: Combine P_t and Q_t (size 2N -> N)
            combined_pool = population + offspring_pop
            metrics_list = [c.metrics for c in combined_pool]
            fronts = fast_non_dominated_sort(metrics_list, self.objectives)

            next_population: List[CandidateConfig] = []
            for rank, front in enumerate(fronts):
                crowding_distances = calculate_crowding_distance(
                    front, metrics_list, self.objectives
                )
                for idx in front:
                    combined_pool[idx].rank = rank
                    combined_pool[idx].crowding_distance = crowding_distances.get(idx, 0.0)

                # Check if entire front fits in next population
                if len(next_population) + len(front) <= pop_size:
                    for idx in front:
                        next_population.append(combined_pool[idx])
                else:
                    # Sort front candidates by crowding distance descending and fill remainder
                    sorted_front = sorted(
                        [combined_pool[idx] for idx in front],
                        key=lambda c: c.crowding_distance,
                        reverse=True,
                    )
                    needed = pop_size - len(next_population)
                    next_population.extend(sorted_front[:needed])
                    break

            population = next_population

        return self.history
