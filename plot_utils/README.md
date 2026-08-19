# Plotting & Visualization Utilities (`plot_utils`)

This directory contains shell scripts and Python modules for generating paper figures, benchmark performance curves, dormant neuron statistics, GPU runtime efficiency plots, and visual attention maps from Weights & Biases (W&B) logs.

---

## ⚠️ Separate Environment Recommended

We **strongly recommend installing plotting dependencies in a separate Python virtual environment** from your primary training environment. 

This prevents dependency version conflicts between RL training packages (e.g., PyTorch, Gym 0.26, Procgen-HD C++ bindings) and plotting packages (e.g., `rliable`, `seaborn`, `matplotlib`, `scipy`).

### 1. Create a Plotting Virtual Environment

```bash
# Create a dedicated virtual environment for plotting
python -m venv rlops_env
source rlops_env/bin/activate
```

### 2. Install Plotting Dependencies

```bash
pip install openrlbenchmark
pip install matplotlib seaborn pandas numpy scipy wandb tyro rliable torchinfo
```

---

## Figure & Table Recreation Map

Each plot script in `plot_utils/` corresponds directly to a benchmark launcher in `benchmark_utils/` for recreating the paper figures:

| TMLR Paper Figure / Table | Description | Benchmark Execution Script | Figure Recreation Plot Script |
|---|---|---|---|
| **Figure 1** | Teaser figure: Parameter count growth vs. resolution (Impala vs Impoola) | — | `plot_utils/plot_network_stats.sh` |
| **Figure 2, 12, 13** | Example Procgen-HD observations across resolutions ((48,48) to (112,112)) | `benchmark_utils/run_saliency.sh` | `plot_utils/plot_saliency.sh` |
| **Figure 4 & 5** | Easy Mode Generalization aggregate performance & width scale ($\tau=2, 3, 4$) | `benchmark_utils/run_easy_mode.sh` | `plot_utils/plot_easy_mode.sh` |
| **Figure 6 & Figure 23** | Sample Efficiency Track (unlimited level distribution) | `benchmark_utils/run_efficiency.sh` | `plot_utils/plot_efficiency.sh` |
| **Figure 7, 21, 22** | Hard Mode Generalization aggregate performance (100M steps) | `benchmark_utils/run_hard_mode.sh` | `plot_utils/plot_hard_mode.sh` |
| **Figure 8, 18, 19, 20** | Per-game environment-level performance breakdown (16 Procgen games) | `benchmark_utils/run_easy_mode.sh` | `plot_utils/plot_easy_mode.sh` |
| **Figure 9 & Figure 16** | Dormant neuron fraction dynamics (Easy/Hard) & layer-wise analysis | `benchmark_utils/run_easy_mode.sh` | `plot_utils/plot_dormant_neurons.sh` |
| **Figure 10 & Figure 11** | Policy/Value saliency mask sparsity & Starpilot visual attention overlays | `benchmark_utils/run_saliency.sh` | `plot_utils/plot_saliency.sh` |
| **Figure 15** | Architectural Ablation Studies ($5\times5$ kernels, $d(z)=50$, DrQ-v2, downsampling) | `benchmark_utils/run_ablations.sh` | `plot_utils/plot_ablations.sh` |
| **Figure 17** | Extreme high-resolution scaling ((128,128) to (224,224)) on Starpilot | `benchmark_utils/run_hard_mode.sh` | `plot_utils/plot_hard_mode.sh` |
| **Figure 24 & Figure 25** | GPU wall-clock training time impact & 5-stage execution time breakdown | `benchmark_utils/run_gpu_timing.sh` | `plot_utils/plot_gpu_timing.sh` |
| **Table 3 & Table 4** | Detailed model architecture summaries (Params, MACs, Memory Footprint) | — | `plot_utils/plot_network_stats.sh` |

---

## Usage

All shell scripts in `plot_utils/` invoke `rlops.py` to fetch logged metrics directly from W&B and generate high-resolution figures in `paper_plots/`.

### Recreating Easy Mode Scaling Figures (Figures 4 & 5)
```bash
bash plot_utils/plot_easy_mode.sh
```

### Recreating Efficiency Track Figure (Figure 6)
```bash
bash plot_utils/plot_efficiency.sh
```

### Recreating Hard Mode Scaling Figures (Figures 7 & 17)
```bash
bash plot_utils/plot_hard_mode.sh
```

### Recreating GPU Runtime Efficiency Figures (Figures 24 & 25)
```bash
bash plot_utils/plot_gpu_timing.sh
```

---

## Script Overview

- `rlops.py`: Customized version of the `openrlbenchmark.rlops` plotting engine included directly in this repository. It contains customized formatting rules, multi-run query filters, custom subplot grid layouts, and enhanced `rliable` Interquartile Mean (IQM) aggregation tailored specifically for Procgen-HD visual scaling benchmarks.
- `plot_utils.py`: Utility functions for metric aggregation, color palette formatting, LaTeX label parsing, and confidence interval estimation using `rliable`.
