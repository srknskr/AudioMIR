# AutoMIR: Multi-Objective Automated Machine Learning for Rhythm Understanding
### Research Report Template (5–8 Pages)

**Author:** Serkan Seker  
**Affiliation:** Music Information Retrieval / Machine Listening  
**Date:** 2026

---

## Abstract
Brief summary (150-250 words) covering motivation, the multi-objective AutoML formulation, search algorithms compared (Random vs TPE vs NSGA-II), key Pareto trade-offs discovered, and findings regarding tempo accuracy versus inference latency.

---

## 1. Introduction & Motivation
- Need for rhythm understanding (tempo estimation, style classification) on edge / low-latency systems.
- Limitations of single-objective optimization (accuracy-only leads to bloated models).
- Bridging evolutionary multi-objective optimization with deep MIR representations.

---

## 2. Research Questions & Hypotheses
- **RQ1**: Can evolutionary multi-objective search discover architectures that reduce latency and model size with negligible accuracy loss?
- **RQ2**: How do Log-Mel, Tempogram, and Dual audio representations compare across tempo and style tasks?
- **RQ3**: Does native NSGA-II outperform Random Search and Optuna TPE under equal evaluation budgets?

---

## 3. Datasets & Leakage Prevention
- Benchmark dataset: Groove MIDI Dataset (official splits).
- Custom dataset: Drum Loop Manifests with `source_id` group-aware partitioning.
- Strict isolation of Test set during AutoML search.

---

## 4. System Architecture & Audio Representations
- Feature extraction: Log-Mel Spectrogram, Fourier Tempogram, Dual representation.
- Neural model families: TinyCNN, CRNN, DualInputNet.
- Multi-task output heads: $\log_2(\text{BPM})$ regression and Style/Meter classification.

---

## 5. Multi-Objective AutoML & NSGA-II Formulation
- Objectives: Maximize Tempo Acc ($\pm 4\%$), Maximize Style Macro-F1, Minimize Median Latency, Minimize Model Size.
- Multi-fidelity training schedule (QUICK, SCREEN, FULL).
- Algorithmic details: Fast non-dominated sorting, crowding distance, binary tournament selection, crossover, and mutation.

---

## 6. Experimental Results & Pareto Analysis
- Pareto front plots (Accuracy vs Latency, Accuracy vs Model Size).
- Hypervolume comparison table across Random, TPE, and Evolutionary algorithms.
- Selected preset models (Best Accuracy, Balanced, Fastest, Smallest).

---

## 7. Ablation Studies
- Representation ablation (Log-Mel vs Tempogram vs Dual).
- Recurrent layer ablation (CNN vs CRNN).
- Domain generalization results on custom loops.

---

## 8. Discussion & Threats to Validity
- Ambiguities in tempo estimation (octave errors / half/double tempo).
- Hardware dependency of latency benchmarks.
- Class distribution imbalances.

---

## 9. Conclusion & Future Work
- Summary of contributions.
- Future extension to beat tracking, downbeat tracking, and Transformer-based backbone search spaces.

---

## References
[1] E. Fonseca et al., "Automated Machine Learning for Audio Classification," 2021.  
[2] K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE TEC, 2002.  
[3] J. Salamon et al., "Music Information Retrieval and Deep Learning," 2018.
