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
    ├── src/
    │   ├── robotics/        -- Primary: Real Trajectory Validation (no training)
    │   ├── foundation/      -- Secondary: Controlled Mechanistic Studies
    │   ├── benchmarks/      -- Tertiary: Downstream Compatibility (training)
    │   └── pilot/           -- Diagnostic: Internal analysis
    └── outputs/             -- Run capsules
"""
