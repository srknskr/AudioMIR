import copy
import random
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from automir.automl.candidate import CandidateConfig


class SearchSpace:
    """Search space manager defining discrete and continuous parameter bounds,

    random sampling, mutation, and crossover operations.
    """

    REPRESENTATIONS = ["logmel", "tempogram", "logmel_tempogram"]
    SEGMENT_DURATIONS = [4.0, 8.0, 12.0]
    N_MELS_CHOICES = [64, 96, 128]
    CONV_BLOCKS_CHOICES = [2, 3, 4]
    BASE_CHANNELS_CHOICES = [16, 32, 64]
    KERNEL_SIZES = [3, 5, 7]
    USE_GRU_CHOICES = [True, False]
    GRU_HIDDEN_CHOICES = [32, 64, 128]
    BATCH_SIZES = [16, 32, 64]

    LR_MIN, LR_MAX = 1e-4, 3e-3
    WD_MIN, WD_MAX = 1e-6, 1e-3
    DROPOUT_MIN, DROPOUT_MAX = 0.10, 0.50

    @classmethod
    def sample_candidate(cls, generation: int = 0) -> CandidateConfig:
        """Sample an independent candidate uniformly at random from the search space."""
        lr_log = np.random.uniform(np.log10(cls.LR_MIN), np.log10(cls.LR_MAX))
        wd_log = np.random.uniform(np.log10(cls.WD_MIN), np.log10(cls.WD_MAX))

        return CandidateConfig(
            candidate_id=str(uuid.uuid4())[:8],
            generation=generation,
            representation=random.choice(cls.REPRESENTATIONS),
            segment_duration=random.choice(cls.SEGMENT_DURATIONS),
            n_mels=random.choice(cls.N_MELS_CHOICES),
            conv_blocks=random.choice(cls.CONV_BLOCKS_CHOICES),
            base_channels=random.choice(cls.BASE_CHANNELS_CHOICES),
            kernel_size=random.choice(cls.KERNEL_SIZES),
            use_gru=random.choice(cls.USE_GRU_CHOICES),
            gru_hidden=random.choice(cls.GRU_HIDDEN_CHOICES),
            dropout=round(float(np.random.uniform(cls.DROPOUT_MIN, cls.DROPOUT_MAX)), 3),
            learning_rate=float(10**lr_log),
            weight_decay=float(10**wd_log),
            batch_size=random.choice(cls.BATCH_SIZES),
        )

    @classmethod
    def crossover(
        cls, parent_a: CandidateConfig, parent_b: CandidateConfig, generation: int = 0
    ) -> CandidateConfig:
        """Uniform crossover combining attributes from two parent candidates."""
        child = copy.deepcopy(parent_a)
        child.candidate_id = str(uuid.uuid4())[:8]
        child.generation = generation
        child.rank = 0
        child.crowding_distance = 0.0
        child.failed = False
        child.failure_reason = None
        child.metrics = {}

        keys = [
            "representation",
            "segment_duration",
            "n_mels",
            "conv_blocks",
            "base_channels",
            "kernel_size",
            "use_gru",
            "gru_hidden",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
        ]

        for k in keys:
            if random.random() > 0.5:
                setattr(child, k, getattr(parent_b, k))

        return child

    @classmethod
    def mutate(
        cls, candidate: CandidateConfig, mutation_prob: float = 0.3
    ) -> CandidateConfig:
        """Mutate parameters of a candidate with given probability per gene."""
        mutant = copy.deepcopy(candidate)
        mutant.candidate_id = str(uuid.uuid4())[:8]
        mutant.rank = 0
        mutant.crowding_distance = 0.0
        mutant.failed = False
        mutant.failure_reason = None
        mutant.metrics = {}

        if random.random() < mutation_prob:
            mutant.representation = random.choice(cls.REPRESENTATIONS)
        if random.random() < mutation_prob:
            mutant.segment_duration = random.choice(cls.SEGMENT_DURATIONS)
        if random.random() < mutation_prob:
            mutant.n_mels = random.choice(cls.N_MELS_CHOICES)
        if random.random() < mutation_prob:
            mutant.conv_blocks = random.choice(cls.CONV_BLOCKS_CHOICES)
        if random.random() < mutation_prob:
            mutant.base_channels = random.choice(cls.BASE_CHANNELS_CHOICES)
        if random.random() < mutation_prob:
            mutant.kernel_size = random.choice(cls.KERNEL_SIZES)
        if random.random() < mutation_prob:
            mutant.use_gru = not mutant.use_gru
        if random.random() < mutation_prob:
            mutant.gru_hidden = random.choice(cls.GRU_HIDDEN_CHOICES)
        if random.random() < mutation_prob:
            mutant.batch_size = random.choice(cls.BATCH_SIZES)

        # Continuous mutations (Gaussian jitter in log / linear space)
        if random.random() < mutation_prob:
            new_dropout = mutant.dropout + np.random.normal(0, 0.05)
            mutant.dropout = round(float(np.clip(new_dropout, cls.DROPOUT_MIN, cls.DROPOUT_MAX)), 3)

        if random.random() < mutation_prob:
            cur_lr_log = np.log10(mutant.learning_rate)
            new_lr_log = cur_lr_log + np.random.normal(0, 0.25)
            new_lr_log = np.clip(new_lr_log, np.log10(cls.LR_MIN), np.log10(cls.LR_MAX))
            mutant.learning_rate = float(10**new_lr_log)

        if random.random() < mutation_prob:
            cur_wd_log = np.log10(mutant.weight_decay)
            new_wd_log = cur_wd_log + np.random.normal(0, 0.3)
            new_wd_log = np.clip(new_wd_log, np.log10(cls.WD_MIN), np.log10(cls.WD_MAX))
            mutant.weight_decay = float(10**new_wd_log)

        return mutant
