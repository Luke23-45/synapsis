"""
SYNAPSE Applied Robotics — M1 Real-World Memory Utility Proof
=============================================================

This package implements the controlled experiment that proves SYNAPSE's
memory operator provides measurable utility when integrated into a
real-world robotics planning pipeline.

Subpackages:
    - core: Experiment configuration + Z-score normalization
    - synapse: SYNAPSE adapter + offline feature caching
    - planner: Multi-condition transformer planner architecture
    - data: HuggingFace LeRobot data pipeline (PushT, ALOHA, xArm)
    - engine: Training, evaluation, rollout, statistical testing
    - reporting: Visualization + report generation
"""

from src.core.config import ExperimentConfig, Condition

__all__ = ["ExperimentConfig", "Condition"]
