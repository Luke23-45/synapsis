"""
NTH-Attention: Neuro-Topological Hysteresis Attention Architecture v2.0

A SOTA robotic memory system for diffusion-based manipulation.

Modules:
- config: Configuration dataclasses
- temporal_sampler: Module 1 - Adaptive Temporal Sampler (TCN + Sinkhorn)
- geometric_projector: Module 2 - Geometric Manifold Projector (Hyperbolic + Diffusion)
- nth_attention: Module 3 - NTH-Attention Transformer (GQA + RoPE)
- vision_encoder: SOTA SigLIP + Language encoding
- action_head: Flow Matching action generation
- planner: Unified NTH Diffusion Planner
- losses: Loss functions
"""

# FIXED: Import config from parent package
from SYNAPSIS.config import NTHConfig, NTHConfigSmall, NTHConfigBase, NTHConfigLarge, get_config

from .temporal_sampler import (
    AdaptiveTemporalSampler,
    StochasticSinkhornTopK,
    TemporalContextNetwork,
    AdaptiveNeuralHysteresis,
)
from .geometric_projector import (
    GeometricManifoldProjector,
    HyperbolicProxyDistance,
    MultiScaleAdaptiveDiffusion,
    FractalInceptionBlock,
)
from .nth_attention import (
    NTHAttentionTransformer,
    NTHTransformerBlock,
    AdaLNZero, # V2 Version
    # NTH_GQA_Attention # Internal
)
from .vision_encoder import (
    VisionLanguageEncoder,
    SigLIPVisionEncoder,
    ProprioceptionEncoder,
    LanguageEncoder,
    FallbackVisionLanguageEncoder,
    create_vision_encoder,
)
from .action_head import (
    FlowMatchingActionHead,
    ActionNormalizer,
    SinusoidalPosEmb,
)
from .planner import NTHDiffusionPlanner, InputValidator, OutputValidator

from .losses import NTHLoss, cosine_temperature_schedule, linear_warmup_cosine_decay

# NOTE: Data and Utils commented out to prevent import errors during migration
# from .data import (
#     NTHDataset,
#     SyntheticNTHDataset,
#     nth_collate_fn,
#     create_nth_dataloader,
#     create_synthetic_dataloader,
# )
# from .utils import count_parameters, get_parameter_groups, initialize_weights, create_dummy_batch

__version__ = "2.0.0"
__all__ = [
    # Config
    "NTHConfig",
    "NTHConfigSmall",
    "NTHConfigBase",
    "NTHConfigLarge",
    "get_config",
    # Module 1
    "AdaptiveTemporalSampler",
    "StochasticSinkhornTopK",
    "TemporalContextNetwork",
    "AdaptiveNeuralHysteresis",
    # Module 2
    "GeometricManifoldProjector",
    "HyperbolicProxyDistance",
    "MultiScaleAdaptiveDiffusion",
    "FractalInceptionBlock",
    # Module 3
    "NTHAttentionTransformer",
    "NTHTransformerBlock",
    "AdaLNZero",
    # Vision + Language
    "VisionLanguageEncoder",
    "SigLIPVisionEncoder",
    "ProprioceptionEncoder",
    "LanguageEncoder",
    "FallbackVisionLanguageEncoder",
    "create_vision_encoder",
    # Action
    "FlowMatchingActionHead",
    "ActionNormalizer",
    "SinusoidalPosEmb",
    # Planner
    "NTHDiffusionPlanner",
    "InputValidator",
    "OutputValidator",
    # Training
    "NTHLoss",
    "cosine_temperature_schedule",
    "linear_warmup_cosine_decay",
]
