# AutoMIR Experimental Protocol & Benchmark Guide

## 1. Research Experiments

### Experiment 1: Equal-Budget Search Strategy Comparison
- **Strategies**: Random Search vs Optuna TPE vs Evolutionary Pareto (NSGA-II)
- **Budget**: Exactly 50 candidate evaluations for each algorithm.
- **Metrics**: Discovered Pareto front cardinality, 2D hypervolume, wall-clock efficiency, best-so-far convergence curve.

### Experiment 2: Feature Representation Ablation
- Compare Log-Mel Spectrogram only vs Tempogram only vs Dual Representation.
- Research question: Does the dual representation provide complementary tempo periodicity information that justifies the extra parameters?

### Experiment 3: Single-Objective vs Multi-Objective Search
- Compare models selected purely for Accuracy vs models discovered on the multi-objective Pareto Front.
- Quantify the latency and model size reductions achievable for < 2% drop in accuracy.

### Experiment 4: Recurrent Temporal Modeling (CNN vs CRNN)
- Evaluate the impact of GRU sequence layers on tempo estimation stability and half/double tempo confusion.

### Experiment 5: Cross-Domain Generalization
- Evaluate models trained on Groove MIDI against custom drum loop archives (`Serkan Loops`).
- Measure domain shift and out-of-distribution robustness.

---

## 2. Command Sequence for Experiments

```bash
# 1. Activate Environment
source .venv/bin/activate

# 2. Run equal-budget comparisons
python scripts/run_search.py --strategy random --evaluations 50 --config configs/standard.yaml
python scripts/run_search.py --strategy tpe --evaluations 50 --config configs/standard.yaml
python scripts/run_search.py --strategy evolutionary --evaluations 50 --config configs/standard.yaml

# 3. Retrain and benchmark final Pareto set on untouched test set
python scripts/retrain_pareto.py --run-id <RUN_ID>
python scripts/benchmark.py --run-id <RUN_ID>
```
