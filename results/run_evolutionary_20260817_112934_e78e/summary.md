# AutoMIR Experiment Run Summary: `run_evolutionary_20260817_112934_e78e`

- **Strategy**: `evolutionary`
- **Dataset**: `synthetic`
- **Date & Time**: `2026-08-17T11:29:34.742803`
- **Git Commit**: `2fdbf9d`
- **Device**: `Apple Silicon (MPS)`
- **Total Evaluations**: 6
- **Successful Evaluations**: 6
- **Pareto Candidates Discovered**: 2
- **Wall-clock Time**: 930.8 seconds

## Discovered Pareto-Optimal Models

| ID | Rep | Blocks | Chans | GRU | Tempo Acc ±4% | Style Macro-F1 | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| `3671db90` | logmel | 4 | 16 | No | 0.0% | 19.1% | 4.06 ms | 2.11 MB |
| `11e4710d` | logmel_tempogram | 3 | 16 | No | 0.0% | 8.3% | 2.73 ms | 0.34 MB |

## Generated Visualizations

- `plots/latency_vs_tempo_acc.png`
- `plots/size_vs_tempo_acc.png`
- `plots/latency_vs_style_f1.png`
- `plots/size_vs_style_f1.png`
