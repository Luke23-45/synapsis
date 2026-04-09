"""
SYNAPSE Empirical Experiments
==============================

Publication-facing experiment suite validating that the SYNAPSE memory
operator provides measurable downstream utility beyond baselines.

Structure
---------
::

    empirical/
    ├── common/              -- Shared baselines, eval, tasks, plotting
    ├── src/                 -- 7 experiment scripts
    │   ├── event_sparse_recovery.py     -- PC-01
    │   ├── memory_sufficiency.py        -- PC-02
    │   ├── topology_value_probe.py      -- PC-03
    │   ├── compression_robustness.py    -- PC-04
    │   ├── public_benchmark.py          -- PC-05
    │   ├── ablation_study.py            -- PC-06
    │   └── pilot_diagnostics.py         -- PC-07
    └── outputs/             -- Run capsules
"""
