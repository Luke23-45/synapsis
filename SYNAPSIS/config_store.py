"""
Hydra ConfigStore Registration

Registers NTHConfig dataclasses with Hydra's ConfigStore API.
This enables type-safe YAML composition with CLI overrides.

Usage:
    # In train.py
    from SYNAPSIS.config_store import register_configs
    register_configs()
    
    @hydra.main(config_path="configs", config_name="config")
    def main(cfg):
        ...
"""

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Import existing dataclasses
from SYNAPSIS.config import (
    NTHConfig,
    NTHConfigSmall,
    NTHConfigBase,
    NTHConfigLarge,
)


def register_configs() -> None:
    """
    Register all configuration schemas with Hydra's ConfigStore.
    
    This must be called BEFORE @hydra.main() is invoked.
    """
    cs = ConfigStore.instance()
    
    # Register base schema (Optional: disabled for more flexible composition)
    # cs.store(name="nth_config_schema", node=NTHConfig)
    
    # Register model presets (Optional: disabled in favor of plain YAMLs for flexibility)
    # cs.store(group="model", name="base", node=NTHConfigBase)
    # cs.store(group="model", name="small", node=NTHConfigSmall)
    # cs.store(group="model", name="large", node=NTHConfigLarge)


def get_config_from_yaml(cfg) -> NTHConfig:
    """
    Convert OmegaConf DictConfig to NTHConfig dataclass with robust hierarchical merging.
    """
    from omegaconf import OmegaConf, MISSING
    from SYNAPSIS.config import NTHConfig
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 1. Start with a fresh NTHConfig
    config_base = NTHConfig()
    
    # 2. Extract and flatten groups hierarchically
    # Priority: root -> models -> projector -> sampler -> training (highest)
    groups_priority = ["model", "projector", "sampler", "training"]
    
    # Convert DictConfig to a resolved dict
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    
    # Build final flat dict
    flat_dict = {}
    
    # A. Add root level primitives first (highest priority for root fields)
    for k, v in cfg_resolved.items():
        if k not in groups_priority and not isinstance(v, dict):
            flat_dict[k] = v
            
    # B. Add groups in priority order
    for group in groups_priority:
        if group in cfg_resolved and isinstance(cfg_resolved[group], dict):
            group_dict = cfg_resolved[group]
            for k, v in group_dict.items():
                # Skip placeholders and missing values
                if v == "???" or v is MISSING or v is None:
                    continue
                # If it's an empty string for a path, don't let it overwrite
                if k in ["train_path", "val_path", "urdf_path", "xml_path"] and not v:
                    continue
                flat_dict[k] = v
                
    # 3. Filter to only NTHConfig fields
    from dataclasses import fields
    valid_fields = {f.name for f in fields(NTHConfig)}
    final_dict = {k: v for k, v in flat_dict.items() if k in valid_fields}
    
    # 4. Final Verification of paths
    for p in ["train_path", "val_path", "urdf_path", "xml_path"]:
        path_val = final_dict.get(p, "")
        if not path_val:
            logger.warning(f"Configuration field '{p}' is empty.")
        else:
            logger.info(f"Resolved {p}: {path_val}")
    
    # Create and validate NTHConfig
    return NTHConfig(**final_dict)


# Auto-register on import (optional, can also call explicitly)
# Uncomment the line below if you want auto-registration
# register_configs()
