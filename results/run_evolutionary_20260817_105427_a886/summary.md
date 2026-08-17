# AutoMIR Experiment Run Summary: `run_evolutionary_20260817_105427_a886`

- **Strategy**: `evolutionary`
- **Dataset**: `synthetic`
- **Date & Time**: `2026-08-17T10:54:27.553120`
- **Git Commit**: `b3165b9`
- **Device**: `Apple Silicon (MPS)`
- **Total Evaluations**: 6
- **Successful Evaluations**: 6
- **Pareto Candidates Discovered**: 3
- **Wall-clock Time**: 157.3 seconds

## Discovered Pareto-Optimal Models

| ID | Rep | Blocks | Chans | GRU | Tempo Acc ±4% | Style Macro-F1 | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| `c1270b2d` | logmel | 4 | 16 | No | 0.0% | 14.3% | 4.53 ms | 2.11 MB |
| `0e2c4f1c` | logmel_tempogram | 2 | 32 | No | 0.0% | 14.3% | 14.23 ms | 0.94 MB |
| `1034c85d` | logmel_tempogram | 3 | 16 | No | 0.0% | 8.3% | 3.82 ms | 0.34 MB |

## Generated Visualizations

- `plots/latency_vs_tempo_acc.png`
- `plots/size_vs_tempo_acc.png`
- `plots/latency_vs_style_f1.png`
- `plots/size_vs_style_f1.png`
