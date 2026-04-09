"""
SYNAPSE Experiments
====================

Self-contained verification framework for the memory operator M.

Everything is inside this directory: configs, scripts, utils,
common infrastructure, verification experiments, and outputs.

Structure
---------
::

    experiments/
    ├── configs/
    │   └── default.yaml            -- YAML experiment config
    │
    ├── scripts/
    │   └── experiment_runner.py     -- CLI: run one or all experiments
    │
    ├── utils/                       -- Shared infrastructure (from GibbsQ)
    │   ├── config.py                -- Typed dataclass config
    │   ├── model_io.py              -- Run capsule management
    │   ├── logging.py               -- Per-run file logging
    │   ├── exporter.py              -- JSONL + CSV export
    │   ├── progress.py              -- tqdm progress bars
    │   └── run_artifacts.py         -- Path helpers
    │
    ├── common/                      -- Shared experiment code
    │   ├── chart_exporter.py        -- Multi-format chart/data export
    │   ├── metrics.py               -- Distance/comparison functions
    │   ├── plotting.py              -- Domain-specific plot functions
    │   ├── report.py                -- Structured reporting
    │   ├── theme.py                 -- Publication matplotlib themes
    │   └── trajectory_generators.py -- Synthetic trajectory factories
    │
    ├── verification/                -- Formal claim verification
    │   ├── causality.py             -- EXP-01: Theorem 8.1
    │   ├── bounded_cardinality.py   -- EXP-02: Theorem 8.2
    │   ├── changepoint_identification.py -- EXP-03: Theorem 8.3
    │   ├── exact_reconstruction.py  -- EXP-04: Corollary 8.4
    │   ├── topological_stability.py -- EXP-05: Theorem 8.5
    │   ├── hysteretic_encoding.py   -- EXP-06: Section 3
    │   ├── information_loss.py      -- EXP-07: Proposition 8.6
    │   └── end_to_end.py            -- EXP-08: Section 8
    │
    └── outputs/                     -- Run capsules go here

Run via
-------
::

    python experiments/scripts/experiment_runner.py all --verbose
    python experiments/scripts/experiment_runner.py causality -v
"""
