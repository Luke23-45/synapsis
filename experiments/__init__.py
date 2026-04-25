"""
SYNAPSE Z2 Experiments
=======================

Verification and empirical validation framework for the Z2 memory operator.

Structure (10 total experiments)
---------------------------------
::

    experiments/
    ├── scripts/
    │   ├── experiment_runner.py     -- CLI: verification experiments
    │   └── empirical_runner.py      -- CLI: empirical experiments
    │
    ├── verification/                -- 6 formal claim verifications
    │   ├── relaxed_selector.py      -- VZ2-01: Prop 5.1, 5.2
    │   ├── hard_projection.py       -- VZ2-02: Prop 6.1, 6.2
    │   ├── exact_recovery.py        -- VZ2-03: Thm 6.3
    │   ├── full_operator.py         -- VZ2-04: Thm 12.1, 12.2
    │   ├── topological_stability.py -- VZ2-05: Thm 12.3
    │   └── sufficiency_metric.py    -- VZ2-06: Prop 12.4, 9.1
    │
    ├── empirical/src/               -- 4 empirical experiments
    │   ├── robotics/
    │   │   ├── anchor_phase_alignment.py  -- EZ2-01
    │   │   └── topology_structure.py      -- EZ2-02
    │   └── foundation/
    │       ├── event_sparse_recovery.py   -- EZ2-03
    │       └── memory_sufficiency.py      -- EZ2-04

Source of truth: docs/formal_math/z2/
"""
