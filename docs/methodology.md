# AutoMIR Research Methodology

## 1. Mathematical Formulation

AutoMIR optimizes neural architectures and audio representations across multiple competing objectives.

Let $\mathcal{C}$ denote the hyperparameter and architectural search space. A candidate $c \in \mathcal{C}$ defines:
- Audio representation $\phi \in \{\text{Log-Mel}, \text{Tempogram}, \text{Dual}\}$
- Model architecture $M(x; \theta_c)$
- Training hyperparameters $\Lambda_c = \{\eta, \lambda_{\text{wd}}, B\}$

### Multi-Objective Problem

$$\max_{c \in \mathcal{C}} \quad \mathbf{F}(c) = \begin{bmatrix} f_{\text{tempo}}(c) \\ f_{\text{style}}(c) \\ -f_{\text{latency}}(c) \\ -f_{\text{size}}(c) \end{bmatrix}$$

Where:
1. $f_{\text{tempo}}(c) = \text{Accuracy}_{\pm 4\%}(y_{\text{bpm}}, \hat{y}_{\text{bpm}})$
2. $f_{\text{style}}(c) = \text{Macro-F1}(y_{\text{style}}, \hat{y}_{\text{style}})$
3. $f_{\text{latency}}(c) = \text{Median Latency (ms at batch size 1)}$
4. $f_{\text{size}}(c) = \text{Serialized model state\_dict size in MB}$

---

## 2. Multi-Task Output Formulations

### Tempo Estimation (Log-Space Regression)
Tempo is modeled by predicting $\hat{z} = \log_2(\text{BPM})$. The predicted linear tempo is recovered via:
$$\hat{\text{BPM}} = 2^{\hat{z}}$$
This formulation stabilizes gradients across wide BPM ranges and gracefully penalizes octave jumps.

### Style & Meter Classification
Outputs logits passed to standard softmax Cross-Entropy loss.

$$\mathcal{L}_{\text{total}} = w_{\text{tempo}} \mathcal{L}_{\text{SmoothL1}}(\hat{z}, z) + w_{\text{style}} \mathcal{L}_{\text{CE}}(\hat{y}_s, y_s) + w_{\text{meter}} \mathcal{L}_{\text{CE}}(\hat{y}_m, y_m)$$

---

## 3. Evolutionary Pareto Optimization (NSGA-II)

AutoMIR provides a native, transparent implementation of the NSGA-II algorithm:
1. **Non-dominated Sorting**: Partitions candidates into Pareto hierarchy fronts $\mathcal{F}_1, \mathcal{F}_2, \dots$.
2. **Crowding Distance**: Preserves diversity along the Pareto frontier by calculating density around each solution.
3. **Tournament Selection**: Chooses individuals prioritizing lower rank, breaking ties with larger crowding distance.
4. **Elitist Survivor Selection**: Combines parent and offspring pools ($P_t \cup Q_t$ of size $2N$) and preserves the top $N$ solutions.

---

## 4. Multi-Fidelity Training Strategy

To evaluate large candidate budgets without excessive compute:
- **QUICK**: 25% data subset, 2-3 epochs (filters out unstable or invalid combinations).
- **SCREEN**: 50% data subset, 5-10 epochs (reliable relative ranking).
- **FULL**: 100% data, 30+ epochs with early stopping (applied only to final selected Pareto-optimal candidates).
