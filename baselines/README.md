# SYNAPSE Phase 4: Cross-Domain Memory Utility Proof

Controlled experiment proving SYNAPSE's non-parametric memory operator `M()` provides measurable utility across **multiple manipulation domains** when integrated into a transformer-based robotics planner.

## What's New in Phase 4

- **Public benchmarks only**: 3 established HuggingFace datasets — fully reproducible, no custom/synthetic data
- **LeRobot adapter**: Direct HuggingFace integration for PushT, ALOHA, and xArm datasets
- **Cross-domain reporting**: MSE matrix heatmaps, per-dataset statistical comparisons, and publication-quality plots
- **Unified adapter architecture**: Pluggable data adapters normalize all formats into a common `RobotEpisode`

## Datasets

All baselines use **public, reproducible** benchmarks. Custom data (Pick & Place) is reserved for the end-to-end deployment section.

| # | Dataset | Source | Episodes | State | Action | Task | Citation |
|---|---------|--------|----------|-------|--------|------|----------|
| D1 | PushT | `lerobot/pusht` | 206 | 2 | 2 | 2D Non-prehensile Pushing | Chi et al., IJRR 2024 |
| D2 | ALOHA Transfer Cube | `lerobot/aloha_sim_transfer_cube_human` | 50 | 14 | 14 | Bimanual Cube Handover | Zhao et al., RSS 2023 |
| D3 | xArm Lift | `lerobot/xarm_lift_medium` | ~100 | 4 | 3 | Single-arm Object Lifting | Hansen et al., 2022 |

## Quick Start

### Smoke Test (~2 minutes)

```bash
python run_experiment.py --config configs/experiment/smoke.yaml
```

### Full Cross-Domain Experiment

```bash
# Downloads into baselines/data/datasets/ by default
# Step 1: Download HuggingFace datasets
python download_datasets.py

# Step 2: Run full experiment (all datasets × all conditions)
python run_experiment.py --config configs/experiment/full.yaml

# Step 3: Run specific datasets or conditions
python run_experiment.py --config configs/experiment/full.yaml --datasets pusht xarm_lift
python run_experiment.py --config configs/experiment/full.yaml --conditions A1_recent B_synapse
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Experimental Design

### Independent Variable: Memory Conditioning Strategy

| Condition | Abbreviation | Memory Input |
|-----------|-------------|--------------|
| Recent Window | A1 | Last W steps of proprioceptive history |
| Uniform Subsampling | A2 | W steps uniformly sampled across full horizon |
| SYNAPSE Full | B | Pre-computed M() features (anchor cloud + topological summary) |
| Anchors Only | B-Anc | M() anchor tokens only (topology zeroed) |
| Topology Only | B-Topo | M() topology token only (anchors zeroed) |

### Dependent Variables

- Action prediction MSE (primary)
- Action cosine similarity
- Phase prediction accuracy (auxiliary)
- Rollout error accumulation

### Statistical Protocol

- **Primary test:** Two-sided Welch's t-test (α=0.05)
- **Effect size:** Cohen's d with bootstrapped 95% CI (10,000 samples)
- **Normality check:** Shapiro-Wilk on paired differences
- **Non-parametric backup:** Wilcoxon signed-rank test
- **Multiple comparison correction:** Bonferroni (across 4 datasets)
- **Seeds:** 5 (primary conditions), 3 (ablation conditions)

## Project Structure

```
baselines/
├── configs/
│   ├── dataset/                    # Per-dataset dimension specs
│   ├── experiment/
│   │   ├── full.yaml               # All datasets × all conditions
│   │   └── smoke.yaml              # Quick validation
│   ├── default.yaml                # Legacy single-dataset config
│   └── *.yaml                      # Legacy ablation configs
├── src/
│   ├── core/
│   │   ├── config.py               # ExperimentConfig + DatasetSpec
│   │   └── normalization.py        # Z-score normalization
│   ├── data/
│   │   ├── adapters/
│   │   │   ├── base_adapter.py     # RobotEpisode + BaseDatasetAdapter
│   │   │   └── lerobot_adapter.py  # HuggingFace LeRobot integration
│   │   ├── dataset.py              # Unified RoboticsDataset
│   │   └── registry.py             # Adapter factory
│   ├── engine/
│   │   ├── train.py                # Training loop
│   │   ├── evaluate.py             # Evaluation engine
│   │   ├── metrics.py              # Statistical testing
│   │   └── rollout.py              # Autoregressive rollout
│   ├── planner/
│   │   ├── planner.py              # Base transformer backbone
│   │   ├── planner_recent.py       # Condition A1
│   │   ├── planner_uniform.py      # Condition A2
│   │   └── planner_synapse.py      # Condition B + ablations
│   ├── synapse/
│   │   ├── synapse_adapter.py      # Feature projection
│   │   └── synapse_cache.py        # Offline M() caching
│   └── reporting/
│       ├── visualize.py            # Publication plots
│       └── report.py               # JSON/Markdown reports
├── tests/
│   ├── test_adapters.py            # Adapter integration tests
│   ├── test_dataset.py
│   ├── test_planner.py
│   └── test_train.py
├── download_datasets.py            # HuggingFace dataset downloader
├── run_experiment.py               # Main entry point
├── run_cache.py                    # SYNAPSE feature caching
└── README.md
```

## Key Design Decisions

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Variable state/action dimensions across datasets | Per-dataset `DatasetSpec` overrides in config; dimension-agnostic adapters |
| 2 | LeRobot datasets lack phase annotations | Synthesized from reward thresholds or episode progress ratio |
| 3 | M() is dimension-agnostic | Uses pairwise distances and topological invariants — works on any D |
| 4 | Cross-contamination of normalization stats | Strictly per-dataset normalization from training split |
| 6 | Format heterogeneity | Unified adapter interface normalizing formats to `RobotEpisode` |

## Citations

```bibtex
@article{chi2024diffusionpolicy,
  author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  journal = {The International Journal of Robotics Research},
  year = {2024},
}

@article{Zhao2023LearningFB,
  title = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author = {Tony Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  journal = {RSS},
  year = {2023},
}
```
