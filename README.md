# AutoMIR: Multi-Objective Automated Machine Learning for Rhythm Understanding

[![CI](https://github.com/srknskr/AutoMIR/actions/workflows/ci.yml/badge.svg)](https://github.com/srknskr/AutoMIR/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)

**AutoMIR** is an open-source, research-grade Music Information Retrieval (MIR) and Machine Listening framework that automates the joint discovery of **audio representations**, **deep neural network architectures**, and **training hyperparameters** for rhythm understanding tasks (Tempo Estimation & Rhythm Style Classification).

Rather than treating model accuracy as an isolated objective, AutoMIR formulates neural architecture search as a **multi-objective Pareto optimization problem**, simultaneously optimizing predictive accuracy, inference latency (ms), and model footprint (MB).

---

## 📑 Table of Contents
- [1. Scientific Motivation & Research Questions](#1-scientific-motivation--research-questions)
- [2. System Architecture](#2-system-architecture)
- [3. Audio Representations & DSP Engine](#3-audio-representations--dsp-engine)
- [4. Deep Learning Model Architectures](#4-deep-learning-model-architectures)
  - [4.1 TinyCNN](#41-tinycnn)
  - [4.2 CRNN (Convolutional Recurrent Neural Network)](#42-crnn-convolutional-recurrent-neural-network)
  - [4.3 DualInputNet (Multi-Modal Fusion)](#43-dualinputnet-multi-modal-fusion)
  - [4.4 Multi-Task Output Heads](#44-multi-task-output-heads)
- [5. Multi-Objective AutoML & NSGA-II Search](#5-multi-objective-automl--nsga-ii-search)
- [6. Datasets & Zero-Leakage Guarantee](#6-datasets--zero-leakage-guarantee)
- [7. Evaluation Metrics & Benchmarks](#7-evaluation-metrics--benchmarks)
- [8. Installation & Environment Setup](#8-installation--environment-setup)
- [9. Command-Line Interface (CLI) Guide](#9-command-line-interface-cli-guide)
- [10. Interactive Streamlit Dashboard](#10-interactive-streamlit-dashboard)
- [11. Reproducibility & Experiment Tracking](#11-reproducibility--experiment-tracking)
- [12. Repository Structure](#12-repository-structure)
- [13. License & Citation](#13-license--citation)

---

## 1. Scientific Motivation & Research Questions

Deep learning models in Music Information Retrieval often suffer from excessive parameter counts and high computational latency, making them impractical for real-time applications such as Digital Audio Workstations (DAWs), live performance plugins, and edge/mobile devices.

AutoMIR investigates the following core research questions:
- **RQ1 (Pareto Frontier Discovery):** *Can automated multi-objective search discover compact, low-latency rhythm models that retain strong predictive accuracy with negligible performance degradation compared to oversized models?*
- **RQ2 (Audio Representations):** *How do Log-Mel Spectrograms, Fourier Tempograms, and Dual-Tower representations compare across continuous tempo regression and discrete rhythm style classification?*
- **RQ3 (Algorithmic Comparison):** *Under identical candidate-evaluation budgets, does native evolutionary NSGA-II discover superior Pareto fronts compared to Random Search and Bayesian Optuna/TPE baselines?*
- **RQ4 (Domain Generalization):** *How robustly do Pareto-optimal models transfer across acoustic domains (e.g., from public MIDI-aligned drums to custom studio drum loop libraries)?*

---

## 2. System Architecture

```
                                  +-----------------------+
                                  |     Audio Stream      |
                                  |  (WAV / MP3 / FLAC)   |
                                  +-----------+-----------+
                                              |
                                  +-----------v-----------+
                                  |  Deterministic Audio  |
                                  |     Preprocessing     |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
         +-----------v-----------+                         +-----------v-----------+
         |  Log-Mel Spectrogram  |                         |   Fourier Tempogram   |
         |  (64, 96, 128 Mel)    |                         |  (Periodicity Bins)   |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                  +-----------v-----------+
                                  |    Candidate Model    |
                                  | TinyCNN / CRNN / Dual |
                                  +-----------+-----------+
                                              |
                             +----------------+----------------+
                             |                                 |
                 +-----------v-----------+         +-----------v-----------+
                 |  Tempo Head log2(BPM) |         | Style / Meter Head    |
                 |  (Log-space SmoothL1) |         | (Softmax CrossEntropy)|
                 +-----------+-----------+         +-----------+-----------+
                             |                                 |
                             +----------------+----------------+
                                              |
                                  +-----------v-----------+
                                  |   Evaluation Engine   |
                                  | Quality, Latency, Size|
                                  +-----------+-----------+
                                              |
                                  +-----------v-----------+
                                  | Multi-Objective AutoML|
                                  | (Random / TPE / NSGA) |
                                  +-----------+-----------+
                                              |
                                  +-----------v-----------+
                                  | 2D/3D Pareto Frontier |
                                  | (Best / Fast / Small) |
                                  +-----------------------+
```

---

## 3. Audio Representations & DSP Engine

AutoMIR provides deterministic audio preprocessing and feature extraction with automatic disk caching:
1. **Log-Mel Spectrogram (`logmel`):** Captures spectral timbre and harmonic distribution across configurable Mel bins ($N_{\text{mels}} \in \{64, 96, 128\}$), with amplitude-to-dB conversion.
2. **Fourier Tempogram (`tempogram`):** Extracts rhythmic periodicity and tempo harmonics by computing the Short-Time Fourier Transform of the onset strength envelope.
3. **Dual Representation (`logmel_tempogram`):** Concatenates both spectral (timbral) and periodicity (rhythmic) features into complementary input streams.
4. **Deterministic SHA-256 Feature Caching:** Hashes audio content identity, sample rate, segment duration, and representation parameters to guarantee stale features are never silently reused.

---

## 4. Deep Learning Model Architectures

```
+-----------------------------------------------------------------------------------+
| Model Family       | Description                        | Best For                |
+--------------------+------------------------------------+-------------------------+
| 1. TinyCNN         | 2D Conv + BatchNorm + ReLU/Dropout | Ultra-low latency & CPU |
| 2. CRNN            | 2D Conv + Bidirectional GRU Layer  | Temporal sequence beats |
| 3. DualInputNet    | Dual-Tower Conv/CRNN + Fusion Dense| Maximum Accuracy/F1     |
+-----------------------------------------------------------------------------------+
```

### 4.1 TinyCNN
- Configurable convolution blocks ($2, 3, 4$).
- Base channels ($16, 32, 64$) doubling with depth.
- Adaptive 2D pooling collapsing spatial dimensions into a compact latent vector.
- Ideal for embedded and edge inference.

### 4.2 CRNN (Convolutional Recurrent Neural Network)
- 2D Conv feature extractor that preserves temporal resolution along the time axis.
- **Bidirectional GRU layer** (hidden dimension $\in \{32, 64, 128\}$) modeling long-range rhythmic dependencies and syncopated accents.
- Temporal average pooling feeding into multi-task heads.

### 4.3 DualInputNet (Multi-Modal Fusion)
- Dedicated convolutional/recurrent tower for Log-Mel spectrograms.
- Dedicated convolutional/recurrent tower for Fourier tempograms.
- Feature fusion layer concatenating representations and projecting through dense non-linear layers.

### 4.4 Multi-Task Output Heads
- **Tempo Regression Head:** Predicts $\hat{z} = \log_2(\text{BPM})$ using Smooth L1 Loss. The predicted linear tempo is recovered via:
  $$\hat{\text{BPM}} = 2^{\hat{z}}$$
  This formulation handles octave jumps and wide tempo ranges smoothly.
- **Rhythm / Style Head:** Multi-class classification predicting genre/style logits (Rock, Funk, Jazz, Latin, etc.) via Cross-Entropy Loss.
- **Meter Head (Optional):** Time-signature classification ($4/4, 3/4, 6/8$) with automatic disablement if class imbalance exceeds thresholds.

---

## 5. Multi-Objective AutoML & NSGA-II Search

AutoMIR implements a fully transparent, native **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**:

1. **Multi-Objective Problem Formulation:**
   $$\max_{c \in \mathcal{C}} \quad \mathbf{F}(c) = \begin{bmatrix} \text{Tempo Accuracy}_{\pm 4\%}(c) \\ \text{Style Macro-F1}(c) \\ -\text{Median Latency}_{\text{ms}}(c) \\ -\text{Model Size}_{\text{MB}}(c) \end{bmatrix}$$

2. **Core Algorithmic Components:**
   - **Fast Non-dominated Sorting:** Partitions candidate populations into Pareto fronts $\mathcal{F}_1, \mathcal{F}_2, \dots, \mathcal{F}_k$.
   - **Crowding Distance Calculation:** Enforces solution diversity along the frontier.
   - **Binary Tournament Selection:** Prefers individuals with lower Pareto rank, breaking ties using crowding distance.
   - **Hyperparameter Crossover & Mutation:** Discrete architectural mutation and continuous Gaussian jitter in log-learning-rate and weight-decay spaces.
   - **Elitist Selection:** $(N + N) \to N$ pool selection guaranteeing best Pareto candidates are preserved across generations.

3. **Baseline Comparison:**
   - **Random Search:** Uniform parameter sampling under equal candidate budget.
   - **Optuna TPE:** Bayesian optimization using Tree-structured Parzen Estimator.

---

## 6. Datasets & Zero-Leakage Guarantee

### Dataset 1: Groove MIDI Dataset (Default Benchmark)
- 1,150+ real drum performances recorded on Roland V-Drums by professional drummers (13.6+ hours).
- Preserves the official Train, Validation, and Test splits.

### Dataset 2: Custom Loop Dataset (`Serkan Loops`)
- User-supplied drum loop collections formatted via CSV manifests:
  ```csv
  audio_path,bpm,meter,genre,source_id
  loops/001.wav,120,4/4,rock,pack01_loop01
  loops/002.wav,120,4/4,rock,pack01_loop01_var2
  loops/003.wav,90,4/4,hiphop,pack03_loop07
  ```
- **Group-Aware Anti-Leakage Protocol:** Strictly guarantees that all variations sharing the same `source_id` remain in exactly ONE split:
  $$\text{Train}_{\text{groups}} \cap \text{Val}_{\text{groups}} = \emptyset, \quad \text{Train}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset, \quad \text{Val}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset$$

### Dataset 3: Synthetic Deterministic Rhythm Generator
- Real-time synthesis of kicks, snares, and hi-hats for instant unit testing and CI without downloading large datasets.

---

## 7. Evaluation Metrics & Benchmarks

| Metric | Category | Description |
|---|---|---|
| **Tempo MAE (BPM)** | Tempo | Mean absolute error in BPM |
| **Tempo Median AE** | Tempo | Median absolute error in BPM |
| **Tempo $\text{Acc}_{\pm 4\%}$** | Tempo | Percentage of predictions within $\pm 4\%$ of ground truth |
| **Tempo $\text{Acc}_{\pm 8\%}$** | Tempo | Percentage of predictions within $\pm 8\%$ of ground truth |
| **Octave-Aware Accuracy** | Tempo | Tolerance-aware accuracy considering $0.5\times, 1.0\times, 2.0\times$ octaves |
| **Half / Double Rate** | Tempo | Ambiguity tracking for half-tempo and double-tempo errors |
| **Macro-F1 (Primary)** | Style / Meter | Class-balanced harmonic mean of precision and recall |
| **Weighted-F1 / Accuracy** | Style / Meter | Global classification accuracy and weighted F1 |
| **Median / p95 Latency** | Efficiency | Per-sample inference time (ms) at batch size = 1 with warm-up |
| **Model Size (MB)** | Efficiency | Serialized `state_dict` disk footprint in megabytes |
| **Parameter Count** | Efficiency | Total learnable weights |

---

## 8. Installation & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/srknskr/AutoMIR.git
cd AutoMIR

# 2. Create Python virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install package and dependencies in editable mode
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Run test suite
pytest -v
```

> **Hardware Acceleration:** AutoMIR automatically detects and utilizes `CUDA` (Nvidia GPUs), `Apple MPS` (Apple Silicon M1/M2/M3/M4), or falls back to `CPU`.

---

## 9. Command-Line Interface (CLI) Guide

### 1. Download Groove MIDI Dataset
```bash
python scripts/download_groove.py
```

### 2. Create Group-Aware Manifest for Custom Loops
```bash
python scripts/create_manifest.py \
  --audio-dir "/path/to/your/drum_loops" \
  --output data/custom_loops_manifest.csv \
  --infer-from-filename
```

### 3. Train Baseline Model
```bash
python scripts/train_baseline.py --config configs/standard.yaml
```

### 4. Run AutoML Searches (Equal-Budget Comparison)
```bash
# Random Search Baseline (50 candidates)
python scripts/run_search.py --strategy random --evaluations 50 --config configs/standard.yaml

# Optuna TPE Bayesian Baseline (50 candidates)
python scripts/run_search.py --strategy tpe --evaluations 50 --config configs/standard.yaml

# Evolutionary Multi-Objective Pareto Search (NSGA-II)
python scripts/run_search.py --strategy evolutionary --evaluations 50 --config configs/standard.yaml
```

### 5. Retrain Pareto Models & Benchmark on Test Set
```bash
# Full-fidelity retraining of discovered Pareto models
python scripts/retrain_pareto.py --run-id <RUN_ID>

# Final test evaluation on untouched Test Set
python scripts/benchmark.py --run-id <RUN_ID>
```

### 6. Reproduce Saved Experiment
```bash
python -m automir.experiments.reproduce <RUN_ID>
```

---

## 10. Interactive Streamlit Dashboard

Launch the live interactive web demo:

```bash
streamlit run dashboard/app.py
```

### Key Features:
- **Audio Upload:** Drag and drop `WAV`, `MP3`, `FLAC`, or `OGG` files (or use built-in synthetic rhythm generator).
- **4 Pareto Presets:**
  - 🏆 **Best Accuracy:** Highest predictive capability.
  - ⚖️ **Balanced:** Optimal trade-off between latency, size, and accuracy.
  - ⚡ **Fastest:** Ultra-low latency model.
  - 🪶 **Smallest:** Minimum memory and disk footprint.
- **Audio Visualizers:** Interactive waveform and dynamic Log-Mel / Tempogram heatmaps.
- **Interactive Pareto Front:** 2D scatter plots (Plotly) exploring multi-objective trade-offs with hover inspections.

---

## 11. Reproducibility & Experiment Tracking

Every experiment is persistently stored in `results/experiments.sqlite` and JSON records:
- Global random seeds for Python, NumPy, and PyTorch.
- Git commit hash at the time of execution.
- Hardware device metadata (processor, GPU/MPS name, OS version).
- Per-candidate hyperparameters and objective evaluation metrics.
- Generated trade-off scatter plots in `results/<RUN_ID>/plots/*.png`.

---

## 12. Repository Structure

```
AutoMIR/
├── .github/workflows/ci.yml           # GitHub Actions CI workflow
├── .gitignore                         # Data, caches, and weight exclusions
├── LICENSE                            # MIT License
├── README.md                          # Comprehensive documentation
├── pyproject.toml                     # Python package metadata
├── requirements.txt                   # Dependency specifications
├── configs/
│   ├── quick.yaml                     # Fast local/CI testing configuration
│   ├── standard.yaml                  # Standard local experiment configuration
│   └── research.yaml                  # High-budget GPU workstation configuration
├── automir/
│   ├── utils/                         # Device selector (CUDA/MPS/CPU) & Seed manager
│   ├── audio/                         # Transforms (Mel, Tempogram) & SHA-256 cache
│   ├── datasets/                      # Groove, Serkan Loops, Synthetic data loaders
│   ├── models/                        # TinyCNN, CRNN, DualInputNet & Multi-Task Heads
│   ├── training/                      # Multi-Task Loss & Multi-Fidelity Trainer
│   ├── evaluation/                    # Tempo & Style metrics, Latency benchmark, Pareto
│   ├── automl/                        # Search Space, Random, TPE, Evolutionary (NSGA-II)
│   ├── experiments/                   # SQLite store, JSON records, Reproduce runner
│   └── inference/                     # Production inference engine for audio files
├── dashboard/
│   └── app.py                         # Streamlit interactive application
├── scripts/
│   ├── download_groove.py             # Groove MIDI downloader
│   ├── create_manifest.py             # Custom manifest generator with anti-leakage
│   ├── train_baseline.py              # Baseline model training CLI
│   ├── run_search.py                  # AutoML search runner CLI
│   ├── retrain_pareto.py              # Full-fidelity Pareto retraining CLI
│   └── benchmark.py                   # Test set evaluation CLI
├── tests/                             # 25 unit and integration tests
├── docs/                              # Methodology, experiments, reproducibility docs
└── results/                           # SQLite database, run directories, checkpoints
```

---

## 13. License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```bibtex
@misc{automir2026,
  author = {Serkan Seker},
  title = {AutoMIR: Multi-Objective Automated Machine Learning for Rhythm Understanding},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/srknskr/AutoMIR}
}
```
