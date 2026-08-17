# AutoMIR Reproducibility Guide

## 1. Determinism and Seeding

AutoMIR enforces reproducibility across all sub-modules via `seed_everything(seed)`:
- Python standard library `random`
- NumPy random generator `np.random.seed`
- PyTorch CPU & CUDA/MPS manual seeds
- Deterministic CuDNN algorithms where supported
- `PYTHONHASHSEED` environment variable

## 2. Leakage Prevention (Group-Aware Splitting)

Audio variations (e.g. loops originating from the same recording session or pack) share common acoustic characteristics. If split randomly, near-duplicate audio leaks into both train and test splits.

AutoMIR solves this with **group-aware splitting**:
- Every audio file must have a `source_id`.
- The dataset partitioner assigns all samples sharing the same `source_id` to exactly ONE split (Train, Validation, or Test).
- Assertions dynamically verify:
  $$\text{Train}_{\text{groups}} \cap \text{Val}_{\text{groups}} = \emptyset, \quad \text{Train}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset, \quad \text{Val}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset$$

## 3. SQLite and Immutable JSON Artifacts

Every search run is logged with:
- Unique Run UUID
- Exact Git commit hash
- Hardware and device metadata (CPU, CUDA device, or Apple Silicon MPS)
- Complete configuration dump
- Per-candidate hyperparameters and objective metrics

To reproduce a completed run:
```bash
python -m automir.experiments.reproduce <RUN_ID>
```
