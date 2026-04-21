"""
Data Integration Layer for NTH-Attention

Adapts the existing expert_dataset.py format for NTH training.
Handles proprioception history extraction and data formatting.

Key Features:
- Wraps ExpertTrajectoryDataset from redhot
- Extracts proprio_history for temporal anchoring
- Handles image preprocessing
- Provides robust collate function
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class NTHDataset(Dataset):
    """
    NTH-compatible dataset wrapper for expert demonstrations.
    
    Wraps the existing ExpertTrajectoryDataset and adds:
    - Proprioception history extraction (for temporal anchoring)
    - Image preprocessing (resize, normalize)
    - Multi-view support (primary, wrist, goal)
    - Action chunk extraction
    
    Args:
        base_dataset: ExpertTrajectoryDataset instance
        proprio_horizon: Number of history timesteps (T)
        action_chunk_size: Number of future actions to predict (K)
        image_size: Target image size (H=W)
        normalize_images: Whether to normalize to ImageNet stats
    """
    
    # ImageNet normalization stats
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        base_dataset: Dataset,
        proprio_horizon: int = 256,
        action_chunk_size: int = 8,
        image_size: int = 224,
        normalize_images: bool = True,
        use_wrist_camera: bool = True,
        use_language_instruction: bool = True,
        use_awr: bool = False,
    ):
        self.base_dataset = base_dataset
        self.proprio_horizon = proprio_horizon
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.use_wrist_camera = use_wrist_camera
        self.use_language_instruction = use_language_instruction
        self.use_awr = use_awr
        
        # Build index mapping
        self._build_index()
    
    def _build_index(self):
        """
        Build index mapping from flat idx to (episode_idx, timestep_idx).
        
        Skips timesteps that don't have enough future actions.
        """
        self.index_map = []
        
        if hasattr(self.base_dataset, 'episode_metadata'):
            # LMDB-based dataset
            for ep_idx, ep_meta in enumerate(self.base_dataset.episode_metadata):
                ep_len = ep_meta['length']
                # Skip last action_chunk_size timesteps (no future actions)
                valid_len = max(0, ep_len - self.action_chunk_size)
                for t in range(valid_len):
                    self.index_map.append((ep_idx, t))
        else:
            # Fallback: assume single episode or use len()
            total_len = len(self.base_dataset)
            valid_len = max(0, total_len - self.action_chunk_size)
            for t in range(valid_len):
                self.index_map.append((0, t))
        
        self.total_samples = len(self.index_map)
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary containing:
            - 'image': (3, H, W) - Primary camera (normalized)
            - 'proprio': (D,) - Current proprioception
            - 'proprio_history': (T, D) - Proprioception history
            - 'action_chunk': (K, D_act) - Future actions
            - 'wrist_image': (3, H, W) - Wrist camera (optional)
            - 'language_instruction': str - Language instruction
            - 'phase_labels': () - Task phase (if available)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        ep_idx, t = self.index_map[idx]
        
        # Get observation at timestep t
        obs = self._get_observation(ep_idx, t)
        
        # Build output dictionary
        output = {}
        
        # === Primary Image ===
        image = self._process_image(obs.get('image_primary', obs.get('image')))
        output['image'] = image
        
        # === Wrist Image (optional) ===
        if self.use_wrist_camera and 'image_wrist' in obs:
            wrist_image = self._process_image(obs['image_wrist'])
            output['wrist_image'] = wrist_image
        
        # === Language Instruction (replaces goal images) ===
        if self.use_language_instruction:
            instruction = self._get_language_instruction(ep_idx)
            output['language_instruction'] = instruction
        
        # === Current Proprioception ===
        proprio = self._get_proprio(ep_idx, t)
        output['proprio'] = torch.from_numpy(proprio.astype(np.float32))
        
        # === Proprioception History ===
        proprio_history = self._get_proprio_history(ep_idx, t)
        output['proprio_history'] = torch.from_numpy(proprio_history.astype(np.float32))
        
        # === Action Chunk ===
        action_chunk = self._get_action_chunk(ep_idx, t)
        output['action_chunk'] = torch.from_numpy(action_chunk.astype(np.float32))
        
        # === Phase Label (if available) ===
        if hasattr(self.base_dataset, 'get_phase'):
            phase = self.base_dataset.get_phase(ep_idx, t)
            output['phase_labels'] = torch.tensor(phase, dtype=torch.long)
        
        # === Advantages (for AWR) ===
        if self.use_awr:
            adv = self._get_advantage(ep_idx, t)
            output['advantages'] = torch.tensor(adv, dtype=torch.float32)
        
        return output
    
    def _get_observation(self, ep_idx: int, t: int) -> Dict[str, np.ndarray]:
        """Get raw observation from base dataset."""
        if hasattr(self.base_dataset, 'get_observation'):
            return self.base_dataset.get_observation(ep_idx, t)
        elif hasattr(self.base_dataset, '__getitem__'):
            # Standard __getitem__ interface
            # M6 FIX: Use episode length from metadata instead of magic number 1000
            ep_len = getattr(self.base_dataset, 'get_episode_length', lambda e: 1000)(ep_idx)
            return self.base_dataset[ep_idx * ep_len + t]
        else:
            raise AttributeError("Base dataset has no get_observation or __getitem__ method")
    
    def _get_proprio(self, ep_idx: int, t: int) -> np.ndarray:
        """Get proprioception at timestep t."""
        if hasattr(self.base_dataset, 'get_proprio'):
            return self.base_dataset.get_proprio(ep_idx, t)
        else:
            obs = self._get_observation(ep_idx, t)
            return obs.get('proprio', obs.get('state', np.zeros(22, dtype=np.float32)))
    
    def _get_proprio_history(self, ep_idx: int, t: int) -> np.ndarray:
        """
        Get proprioception history ending at timestep t.
        
        If not enough history, pad with first observation.
        """
        if hasattr(self.base_dataset, 'get_proprio_sequence'):
            # Optimized bulk retrieval
            start_t = max(0, t - self.proprio_horizon + 1)
            proprio_seq = self.base_dataset.get_proprio_sequence(ep_idx, start_t, t + 1)
        else:
            # Manual collection
            proprio_list = []
            for i in range(max(0, t - self.proprio_horizon + 1), t + 1):
                proprio_list.append(self._get_proprio(ep_idx, i))
            proprio_seq = np.stack(proprio_list, axis=0)
        
        # Pad if necessary (M5 FIX: use zero-padding instead of repeating first obs)
        if len(proprio_seq) < self.proprio_horizon:
            pad_len = self.proprio_horizon - len(proprio_seq)
            pad = np.zeros((pad_len, proprio_seq.shape[1]), dtype=proprio_seq.dtype)
            proprio_seq = np.concatenate([pad, proprio_seq], axis=0)
        
        return proprio_seq
    
    def _get_action_chunk(self, ep_idx: int, t: int) -> np.ndarray:
        """Get action chunk starting at timestep t."""
        if hasattr(self.base_dataset, 'get_action_chunk'):
            return self.base_dataset.get_action_chunk(ep_idx, t, self.action_chunk_size)
        else:
            # Manual collection
            actions_list = []
            for i in range(t, t + self.action_chunk_size):
                action = self._get_action(ep_idx, i)
                actions_list.append(action)
            return np.stack(actions_list, axis=0)
    
    def _get_action(self, ep_idx: int, t: int) -> np.ndarray:
        """Get single action at timestep t."""
        if hasattr(self.base_dataset, 'get_action'):
            return self.base_dataset.get_action(ep_idx, t)
        else:
            obs = self._get_observation(ep_idx, t)
            return obs.get('action', np.zeros(8, dtype=np.float32))
    
    def _get_advantage(self, ep_idx: int, t: int) -> float:
        """Get advantage for specific timestep."""
        if hasattr(self.base_dataset, 'get_advantage'):
            return self.base_dataset.get_advantage(ep_idx, t)
        else:
            obs = self._get_observation(ep_idx, t)
            return float(obs.get('advantages', 0.0))
    
    def _get_language_instruction(self, ep_idx: int) -> str:
        """
        Get language instruction for episode.
        
        SOTA approach: language instructions replace goal images.
        Much more practical for real-world deployment.
        
        [FIXED] For pick-and-place task, we use a descriptive instruction
        that provides semantic grounding for the model.
        """
        if hasattr(self.base_dataset, 'get_language_instruction'):
            return self.base_dataset.get_language_instruction(ep_idx)
        elif hasattr(self.base_dataset, 'episode_metadata'):
            # Try to get from metadata
            ep_meta = self.base_dataset.episode_metadata[ep_idx]
            if 'language_instruction' in ep_meta:
                return ep_meta['language_instruction']
            if 'task_description' in ep_meta:
                return ep_meta['task_description']
        
        # [FIXED] Default instruction for pick-and-place task
        # This provides semantic grounding for language-conditioned policy learning
        return "Pick up the object and place it on the target location"
    
    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Steps:
        1. Resize to target size
        2. Convert to float [0, 1]
        3. Normalize with ImageNet stats (optional)
        4. Convert to torch tensor (C, H, W)
        """
        # Ensure numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Resize if needed
        H, W = image.shape[:2]
        if H != self.image_size or W != self.image_size:
            if HAS_CV2:
                image = cv2.resize(image, (self.image_size, self.image_size))
            else:
                # Fallback using torch
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                img_tensor = F.interpolate(
                    img_tensor, size=(self.image_size, self.image_size), mode='bilinear'
                )
                image = img_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize_images:
            image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return image_tensor


def nth_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for NTH DataLoader.
    
    Handles variable presence of optional keys (wrist_image, goal_image).
    """
    output = {}
    
    # Required keys
    required_keys = ['image', 'proprio', 'proprio_history', 'action_chunk']
    for key in required_keys:
        output[key] = torch.stack([sample[key] for sample in batch], dim=0)
    
    # Optional tensor keys (stack along batch)
    optional_tensor_keys = ['wrist_image', 'phase_labels', 'advantages']
    for key in optional_tensor_keys:
        if key in batch[0]:
            output[key] = torch.stack([sample[key] for sample in batch], dim=0)
    
    # Language instructions (keep as list of strings)
    if 'language_instruction' in batch[0]:
        output['language_instruction'] = [sample['language_instruction'] for sample in batch]
    
    return output


def create_nth_dataloader(
    base_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    proprio_horizon: int = 256,
    action_chunk_size: int = 8,
    image_size: int = 224,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create NTH DataLoader from base dataset.
    
    Args:
        base_dataset: ExpertTrajectoryDataset or similar
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        proprio_horizon: History length for temporal anchoring
        action_chunk_size: Number of future actions
        image_size: Target image size
        **kwargs: Additional arguments for NTHDataset
    
    Returns:
        DataLoader configured for NTH training
    """
    nth_dataset = NTHDataset(
        base_dataset=base_dataset,
        proprio_horizon=proprio_horizon,
        action_chunk_size=action_chunk_size,
        image_size=image_size,
        **kwargs,
    )
    
    return DataLoader(
        nth_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=nth_collate_fn,
        pin_memory=True,
        drop_last=True,  # Ensure consistent batch sizes
    )


class SyntheticNTHDataset(Dataset):
    """
    Synthetic dataset for testing NTH pipeline without real data.
    
    Generates random but correctly shaped data for:
    - Development and debugging
    - Integration testing
    - Benchmarking
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        proprio_dim: int = 22,
        action_dim: int = 8,
        proprio_horizon: int = 256,
        action_chunk_size: int = 8,
        image_size: int = 224,
        num_phases: int = 5,
    ):
        self.num_samples = num_samples
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.proprio_horizon = proprio_horizon
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.num_phases = num_phases
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate random sample with language instruction."""
        # Language instructions (list of sample task descriptions)
        instructions = [
            "pick up the red block and place it on the blue target",
            "grasp the cup and move it to the tray",
            "push the object to the goal position",
            "stack the blocks in the correct order",
            "close the drawer completely",
        ]
        
        return {
            'image': torch.randn(3, self.image_size, self.image_size),
            'wrist_image': torch.randn(3, self.image_size, self.image_size),
            'language_instruction': instructions[idx % len(instructions)],
            'proprio': torch.randn(self.proprio_dim),
            'proprio_history': torch.randn(self.proprio_horizon, self.proprio_dim),
            'action_chunk': torch.randn(self.action_chunk_size, self.action_dim),
            'phase_labels': torch.randint(0, self.num_phases, ()),
        }


def create_synthetic_dataloader(
    num_samples: int = 1000,
    batch_size: int = 32,
    **kwargs,
) -> DataLoader:
    """Create synthetic dataloader for testing."""
    dataset = SyntheticNTHDataset(num_samples=num_samples, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=nth_collate_fn,
        num_workers=0,
    )
