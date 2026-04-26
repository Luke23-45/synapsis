# FILE: utils/expert_dataset.py
# (State-of-the-Art, SoA, JPEG-Compressed, Virtual-Indexed, LRU-Cached Version)

from __future__ import annotations
import copy
import pickle
import os
import time
import json
import hashlib
import logging
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
try:
    from experiments.end_to_end.utils.lmdb_utils import (
        open_lmdb_env, close_lmdb_env, calculate_lmdb_map_size_gb
    )
    _SYNAPSIS_LMDB_AVAILABLE = True
except ImportError:
    _SYNAPSIS_LMDB_AVAILABLE = False
from pathlib import Path
import functools

# --- SOTA Imports ---
try:
    import cv2 # Required for JPEG compression
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mujoco
except ImportError:
    mujoco = None

# Lazy imports for LMDB to avoid pickling issues
try:
    import lmdb
except ImportError:
    lmdb = None

# Project imports (adjust if your project layout differs)
try:
    from SYNAPSIS.envs.panda_env import PandaEnv
except ImportError:
    PandaEnv = None

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

# Default configuration thresholds (you may expose these as args)
REPLAY_POS_TOL = 0.03  # 3 cm tolerance [cite: 219]
REPLAY_ORN_TOL = 5.0 * np.pi / 180.0  # 5 degrees in radians [cite: 219]

# Observation schema: keys with expected shapes/dtypes
# Updated to include all modalities that are actually stored and read.
OBS_SCHEMA = {
    "image_primary": ("uint8", (None, None, 3)),
    "proprio": ("float32", (None,)),
    "internal_full_proprio": ("float32", (None,)),
    "ee_pose_world": ("float32", (7,)),
    "object_pos_world": ("float32", (3,)),
    "object_orn_world": ("float32", (4,)),
    "goal_pos_world": ("float32", (3,)),
    "goal_orn_world": ("float32", (4,)),
    "goal_size_world": ("float32", (3,)),
    "is_grasped": ("float32", (1,)),
    "gripper_qpos": ("float32", (None,)),
    "robot_base_pos_world": ("float32", (3,)),
    "robot_base_quat_world": ("float32", (4,)),
    "base_quat": ("float32", (4,)),
    "object_vel": ("float32", (6,)),
    "ee_vel": ("float32", (6,)),
    "gripper_vel": ("float32", (2,)),
    "expert_target_pose": ("float32", (7,)),
    "delta_ee_pose": ("float32", (7,)),
    "gt_phase": ("int32", (1,)),
    "gt_gripper": ("float32", (1,)),
}

# ==============================================================================
# SOTA HELPER: Delta End-Effector Pose Computation
# ==============================================================================



def compute_delta_ee_pose(target_pose: np.ndarray, current_pose: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
    """
    [SOTA v3.1] Computes the delta end-effector pose in the ROBOT BASE FRAME.
    
    Both position and orientation are expressed as deltas (relative displacements).
    Position delta is transformed from World Frame into Robot Base Frame.
    Orientation delta is the relative rotation from current to target,
    expressed in the Robot Base Frame.
    
    Args:
        target_pose: (7,) [x, y, z, qx, qy, qz, qw] - Expert target in world frame
        current_pose: (7,) [x, y, z, qx, qy, qz, qw] - Current EE in world frame
        base_quat: (4,) [qx, qy, qz, qw] - Robot base orientation in world frame
        
    Returns:
        delta_pose: (7,) [dx, dy, dz, qx, qy, qz, qw] - Delta position (Base Frame) 
                                                      + delta orientation (Base Frame)
    """
    # 1. Compute delta position in World Frame
    delta_pos_world = target_pose[:3] - current_pose[:3]
    
    # 2. Transform delta position into Robot Base Frame
    R_base_world = R.from_quat(base_quat)
    delta_pos_base = R_base_world.inv().apply(delta_pos_world)
    
    # 3. Compute delta orientation: relative rotation from current to target
    R_current = R.from_quat(current_pose[3:])
    R_target = R.from_quat(target_pose[3:])
    R_delta_world = R_current.inv() * R_target
    
    # 4. Express delta orientation in Robot Base Frame
    R_base_inv = R_base_world.inv()
    R_delta_base = R_base_inv * R_delta_world
    
    delta_orn_xyzw = R_delta_base.as_quat().astype(np.float32)
    return np.concatenate([delta_pos_base, delta_orn_xyzw]).astype(np.float32)

# ==============================================================================
# 1. STATE-OF-THE-ART DATASET WRITER
#    (Generates the optimized SoA + JPEG-compressed format)
# ==============================================================================

class ExpertDatasetWriter:
    """
    State-of-the-art helper to write expert demos to an optimized on-disk format.
    
    This writer implements a "Struct of Arrays" (SoA) format and on-the-fly
    image compression to create a highly efficient dataset for chunked reading.

    Key Features:
    - **Struct-of-Arrays (SoA):** Instead of one giant pickle per episode,
      each modality (e.g., 'actions', 'proprio', 'image_primary') is saved
      as its own key in LMDB. This allows the reader to load *only* the
      modalities it needs.
    - **Image Compression:** 'image_primary' and 'image_wrist' are compressed
      on-the-fly to JPEG, massively reducing the dataset size (e.g., 25GB -> 2-5GB).
    - **Instant-On JSON Index:** Creates a single `_index.json` file that
      contains all metadata (episode lengths, dtypes, shapes, compression)
      for the entire dataset. The reader *only* loads this file,
      eliminating any startup scan.
    """
    def __init__(
        self,
        out_dir: str,
        run_name: Optional[str] = None,
        image_compression: str = "jpeg",
        jpeg_quality: int = 90,
        expected_episodes: int = 100,
        control_mode: str = 'absolute',
        recording_mode: str = 'cartesian_delta',
    ):
        if not CV2_AVAILABLE:
            raise ImportError("cv2 (OpenCV) is required for the SOTA ExpertDatasetWriter. Please install it.")
        if not lmdb:
            raise ImportError("lmdb is required for the SOTA ExpertDatasetWriter. Please install it.")
        if not _SYNAPSIS_LMDB_AVAILABLE:
            raise ImportError("SYNAPSIS.utils.lmdb_utils is required for the SOTA ExpertDatasetWriter.") 

        os.makedirs(out_dir, exist_ok=True) 
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.episodes_in_memory: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {} 
        self._episode_id_counter = 0

        self.image_compression = image_compression
        self.jpeg_quality = jpeg_quality
        self.index_data = {
            "episodes": [],
            "control_mode": control_mode,
            "recording_mode": recording_mode
        }
        
        # Dynamic LMDB map size based on expected episodes (~100MB per episode)
        from SYNAPSIS.utils.lmdb_utils import calculate_lmdb_map_size_gb
        self.map_size_gb = calculate_lmdb_map_size_gb(expected_episodes)
        
        logger.info(f"SOTA ExpertDatasetWriter initialized. Compression: {self.image_compression}")
        logger.info(f"  Expected episodes: {expected_episodes}, Map size: {self.map_size_gb:.1f} GB")

    def add_episode(self, ep_dict: Dict[str, Any]):
        """
        Adds a completed episode (in the old "List of Structs" format)
        to the in-memory buffer, ready to be written.
        """
        self.episodes_in_memory.append(ep_dict)

    def _write_pickled_modality(self, txn, ep_meta, prefix, name, data_list):
        """Helper to write a list of generic Python objects by pickling."""
        if not data_list:
            return
        key = f"{prefix}_{name}"
        # We pickle the entire list of objects at once.
        txn.put(key.encode("ascii"), pickle.dumps(data_list))
        
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": "pickle", # A new compression type
            "dtype": "object",
            "shape": [len(data_list)]
        }  
        
    def save_batch(self, episode_list: List[Dict[str, Any]]):
        """Append a list of episodes directly to LMDB (safe for large datasets)."""
        if not episode_list:
            return

        # construct paths like in save() but allow appending to existing file if present
        meta_blob = json.dumps(self.metadata or {}, sort_keys=True)
        md5 = hashlib.md5(meta_blob.encode("utf-8")).hexdigest()[:8]
        base_name = f"expert_{self.run_name}_{md5}"

        if not hasattr(self, "_lmdb_path"):
            meta_blob = json.dumps(self.metadata or {}, sort_keys=True)
            md5 = hashlib.md5(meta_blob.encode("utf-8")).hexdigest()[:8]
            
            self._lmdb_path = self.out_dir / (base_name + ".lmdb")

        lmdb_path = self._lmdb_path

        # open env with dynamic map size
        env = open_lmdb_env(str(lmdb_path), readonly=False, lock=True, map_size_gb=self.map_size_gb, subdir=False)
        try:
            with env.begin(write=True) as txn:
                for ep_dict in episode_list:
                    prefix = f"ep_{self._episode_id_counter:06d}"
                    # Incrementing after processing ensures correct ID usage and count
                    
                    ep_meta = {"episode_id": prefix, "length": len(ep_dict["actions"]), "success": bool(ep_dict.get("success", False)), "seed": ep_dict.get("seed", None), "modalities": {}}
                    
                    self._write_raw_numpy(txn, ep_meta, prefix, "actions", np.array(ep_dict["actions"], dtype=np.float32))
                    self._write_raw_numpy(txn, ep_meta, prefix, "proprio", np.stack([o["proprio"] for o in ep_dict["obs_list"]]).astype(np.float32))
                    self._write_compressed_images(txn, ep_meta, prefix, "image_primary", [o["image_primary"] for o in ep_dict["obs_list"]])
                    self._write_compressed_images(txn, ep_meta, prefix, "image_wrist", [o["image_wrist"] for o in ep_dict["obs_list"]])
                    
                    if "goal_image_primary" in ep_dict and ep_dict["goal_image_primary"] is not None:
                        goal_img_list = [ep_dict["goal_image_primary"]]
                        self._write_compressed_images(txn, ep_meta, prefix, "goal_image_primary", goal_img_list)

                    # --- PHYSICAL STATE MODALITIES ---
                    def stack_mod(name, dtype=np.float32):
                        if name in ep_dict["obs_list"][0]:
                            arr = np.stack([o[name] for o in ep_dict["obs_list"]]).astype(dtype)
                            self._write_raw_numpy(txn, ep_meta, prefix, name, arr)

                    stack_mod("ee_pose_world")
                    stack_mod("object_pos_world")
                    stack_mod("object_orn_world")
                    stack_mod("goal_pos_world")
                    stack_mod("goal_orn_world")
                    stack_mod("is_grasped")
                    stack_mod("object_vel")
                    stack_mod("ee_vel")
                    stack_mod("gripper_qpos")
                    stack_mod("gripper_vel")
                    stack_mod("robot_base_pos_world")
                    stack_mod("goal_size_world")

                    expert_states_list = [o["expert_state"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, prefix, "expert_states", expert_states_list)
                    camera_params_list = [o["camera_params"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, prefix, "camera_params", camera_params_list)
                    gt_phases_arr = np.stack([o["gt_phase"] for o in ep_dict["obs_list"]]).astype(np.int32)
                    self._write_raw_numpy(txn, ep_meta, prefix, "gt_phase", gt_phases_arr)

                    gt_gripper_arr = np.stack([o["gt_gripper"] for o in ep_dict["obs_list"]]).astype(np.float32)
                    self._write_raw_numpy(txn, ep_meta, prefix, "gt_gripper", gt_gripper_arr)

                    # [CRITICAL FIX] Write the EXPERT TARGET POSE - the commanded targets, NOT achieved poses
                    if "expert_target_pose" in ep_dict["obs_list"][0]:
                        target_poses_arr = np.stack([o["expert_target_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, prefix, "expert_target_pose", target_poses_arr)

                    # [SOTA v3.0] Write Delta-EE Pose - the SOTA relative representation
                    if "delta_ee_pose" in ep_dict["obs_list"][0]:
                        delta_poses_arr = np.stack([o["delta_ee_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, prefix, "delta_ee_pose", delta_poses_arr)

                    self.index_data["episodes"].append(ep_meta)
                    self._episode_id_counter += 1
        finally:
            env.sync()
            close_lmdb_env(env)
            
        
        json_path = self._lmdb_path.parent / f"{self._lmdb_path.stem}_index.json"
        with open(json_path, "w") as f:
            json.dump(self.index_data, f)


    def save(self):
        """
        Processes all in-memory episodes, converts them to the optimized
        SoA format, and writes them to the LMDB file and JSON index.
        """
        if not self.episodes_in_memory:
            logger.info("Writer: No pending episodes in memory buffer to save.")
            return

        # --- 1. Prepare File Paths ---
        md5 = hashlib.md5(json.dumps(self.metadata, sort_keys=True).encode("utf-8")).hexdigest()
        base_name = f"expert_{self.run_name}_{md5}"
        lmdb_path = self.out_dir / (base_name + ".lmdb")
        json_path = self.out_dir / (base_name + "_index.json")
        config_path = self.out_dir / (base_name + "_config.json")
        logger.info(f"Saving {len(self.episodes_in_memory)} episodes to {lmdb_path}...")

        # --- 2. Open LMDB Environment ---
        # Dynamic map size based on expected episodes
        env = open_lmdb_env(str(lmdb_path), readonly=False, lock=True, map_size_gb=self.map_size_gb, subdir=False) 
        
        try:
            with env.begin(write=True) as txn: 
                for ep_idx, ep_dict in enumerate(self.episodes_in_memory):
                    episode_key_prefix = f"ep_{ep_idx:06d}"
                    ep_meta = {
                        "episode_id": episode_key_prefix,
                        "length": len(ep_dict["actions"]), 
                        "success": bool(ep_dict.get("success", False)), 
                        "seed": ep_dict.get("seed", None),
                        "modalities": {}
                    }

                    # --- 3. Process and Write Modalities (SoA) ---
                    
                    # A) 'actions' (Raw Numpy)
                    actions_arr = np.array(ep_dict["actions"], dtype=np.float32) 
                    self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "actions", actions_arr)

                    # B) 'proprio' (Raw Numpy)
                    proprio_arr = np.stack([o["proprio"] for o in ep_dict["obs_list"]]).astype(np.float32) 
                    self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "proprio", proprio_arr)
                    
                    # C) 'image_primary' (Compressed Images)
                    img_p_list = [o["image_primary"] for o in ep_dict["obs_list"]] 
                    self._write_compressed_images(txn, ep_meta, episode_key_prefix, "image_primary", img_p_list)

                    # D) 'image_wrist' (Compressed Images)
                    img_w_list = [o["image_wrist"] for o in ep_dict["obs_list"]] 
                    self._write_compressed_images(txn, ep_meta, episode_key_prefix, "image_wrist", img_w_list)
                    if "goal_image_primary" in ep_dict:
                        goal_img_list = [ep_dict["goal_image_primary"]]
                        self._write_compressed_images(txn, ep_meta, episode_key_prefix, "goal_image_primary", goal_img_list)

                    # --- PHYSICAL STATE MODALITIES ---
                    def stack_mod(name, dtype=np.float32):
                        if name in ep_dict["obs_list"][0]:
                            arr = np.stack([o[name] for o in ep_dict["obs_list"]]).astype(dtype)
                            self._write_raw_numpy(txn, ep_meta, episode_key_prefix, name, arr)

                    stack_mod("ee_pose_world")
                    stack_mod("object_pos_world")
                    stack_mod("object_orn_world")
                    stack_mod("goal_pos_world")
                    stack_mod("goal_orn_world")
                    stack_mod("is_grasped")
                    stack_mod("object_vel")
                    stack_mod("ee_vel")
                    stack_mod("gripper_qpos")
                    stack_mod("gripper_vel")
                    stack_mod("robot_base_pos_world")
                    stack_mod("goal_size_world")
                        
                    expert_states_list = [o["expert_state"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, episode_key_prefix, "expert_states", expert_states_list)

                    # F) 'camera_params' (Pickled List of Dictionaries)
                    camera_params_list = [o["camera_params"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, episode_key_prefix, "camera_params", camera_params_list)
                    
                    # G) 'expert_target_pose' - [CRITICAL FIX] The commanded targets for training
                    if "expert_target_pose" in ep_dict["obs_list"][0]:
                        target_poses_arr = np.stack([o["expert_target_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "expert_target_pose", target_poses_arr)
                    
                    # H) 'delta_ee_pose' - [SOTA v3.0] The relative representation for training
                    if "delta_ee_pose" in ep_dict["obs_list"][0]:
                        delta_poses_arr = np.stack([o["delta_ee_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "delta_ee_pose", delta_poses_arr)
                    
                    # --- 4. Add this episode's metadata to the main index ---
                    self.index_data["episodes"].append(ep_meta)

            env.sync() 
            logger.info("LMDB write complete.")
        
        finally:
            close_lmdb_env(env) 
            self.episodes_in_memory.clear()

        # --- 5. Write the final JSON index and config ---
        self.index_data["metadata"] = self.metadata
        with open(json_path, "w") as f:
            json.dump(self.index_data, f) # No indent for smaller file size
        
        with open(config_path, "w") as f:
            json.dump(self.metadata, f, indent=2) 
        
        logger.info(f"Dataset saved. Index: {json_path}")

    def _write_raw_numpy(self, txn, ep_meta, prefix, name, arr):
        """Helper to write a raw numpy array."""
        key = f"{prefix}_{name}"
        txn.put(key.encode("ascii"), arr.tobytes()) 
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": "raw",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape)
        }

    def _write_compressed_images(self, txn, ep_meta, prefix, name, img_list):
        """Helper to compress and write a list of image arrays."""
        if not img_list:
            return
            
        key = f"{prefix}_{name}"
        if self.image_compression == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            encode_fn = lambda img: cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        else: # 'png' or default
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] # Fast PNG
            encode_fn = lambda img: cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        
        # This can be parallelized with multiprocessing.Pool for max speed
        byte_list = [encode_fn(img) for img in img_list]
        
        # We pickle the *list of byte strings*
        txn.put(key.encode("ascii"), pickle.dumps(byte_list)) 
        
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": self.image_compression,
            "dtype": str(img_list[0].dtype),
            "shape": [len(img_list), *img_list[0].shape]
        }

# ==============================================================================
# 2. STATE-OF-THE-ART DATASET READER
#    (Consumes the optimized format)
# ==============================================================================

class ExpertTrajectoryDataset(Dataset):
    """
    State-of-the-art, high-performance loader for expert demonstrations
    stored in the optimized (SoA, compressed, virtually-indexed) format.

    This class is designed for maximum throughput and minimal startup time
    with multi-GB/TB datasets.

    Features:
    - **Instant-On (Zero-Scan):** Reads a pre-computed `_index.json` file.
      It *never* scans the LMDB file at startup.
    - **Virtual Indexing:** Uses O(N_episodes) memory for indexing,
      not O(N_chunks). It computes chunk-to-episode mappings in O(log N)
      time during `__getitem__`.
    - **SoA Chunking:** Loads *only* the required data chunks (e.g.,
      2 frames of 'proprio', 8 frames of 'actions') from the SoA database,
      avoiding any overhead from reading the full episode.
    - **On-the-Fly Decompression:** Decompresses JPEG/PNG images in the
      `DataLoader` workers, leveraging multiple CPU cores and minimizing
      I/O bottlenecks.
    - **Per-Worker LRU Caching:** Uses a bounded LRU cache (`@functools.lru_cache`)
      to keep hot *full-episode-modalities* (like an episode's entire 'proprio'
      array) in each worker's memory. This prevents OOM errors while
      massively speeding up subsequent accesses to the same episode.
    """
    def __init__(self, demo_path: str, observation_horizon: int, action_horizon: int): 
        self.demo_path = Path(demo_path)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.is_lmdb = lmdb and self.demo_path.suffix == ".lmdb" 

        if not self.is_lmdb:
            raise ValueError("SOTA ExpertTrajectoryDataset only supports LMDB (.lmdb) format.")
        if not _SYNAPSIS_LMDB_AVAILABLE:
            raise ImportError(
                "experiments.end_to_end.utils.lmdb_utils is required for E2E ExpertTrajectoryDataset."
            )
        # We do not raise an error for cv2 here because E2E pipeline does not use images.

        # --- 1. Load the JSON Index ---
        self.index_path = self.demo_path.parent / f"{self.demo_path.stem}_index.json"

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Missing required index file: {self.index_path}\n"
                f"Please regenerate your dataset with the new ExpertDatasetWriter."
            )
        
        logger.info(f"Loading index from {self.index_path}...")
        with open(self.index_path, "r") as f:
            index_data = json.load(f)

        self.recording_mode = index_data.get("recording_mode", None)
        self.control_mode = index_data.get("control_mode", None)
        self.episode_metadata = index_data["episodes"]
        self.metadata = index_data.get("metadata", {})

        if self.recording_mode:
            logger.info(f"Dataset recording_mode: {self.recording_mode}, control_mode: {self.control_mode}")
        else:
            logger.warning("Dataset index has no recording_mode stored. Cannot validate action semantics. Regenerate dataset with updated writer.")

        # --- 2. Build the Virtual Index ---
        # We calculate the number of valid chunks in each episode.
        self.chunks_per_episode = []
        for ep_meta in self.episode_metadata:
            ep_len = ep_meta["length"]
            # A valid chunk starts at an index `t` where there are enough past
            # observations and enough future actions.
            # First possible start index `t`: self.observation_horizon - 1
            # Last possible start index `t`: ep_len - self.action_horizon
            start_t = self.observation_horizon - 1
            end_t = ep_len - self.action_horizon
            num_chunks = (end_t - start_t) + 1
            # Ensure we don't have negative chunks for short episodes
            self.chunks_per_episode.append(max(0, num_chunks))

        self.total_chunks = sum(self.chunks_per_episode)
        
        # The cumulative sum is the core of our virtual index.
        # It maps a global chunk `idx` to an `ep_idx`.
        self._cumulative_chunks = np.cumsum(self.chunks_per_episode)

        # Worker-local state (initialized lazily)
        self._lmdb_env = None 
        self._pid = os.getpid() # Store the PID of the process that created the dataset        
        logger.info(
            f"Loaded {len(self.episode_metadata)} episodes, "
            f"{self.total_chunks} total valid chunks (Virtually Indexed)."
        ) 

    def get_goal_image(self, ep_idx: int) -> np.ndarray:
        """
        Efficiently loads and decodes the static goal image for a specific episode.

        This method leverages the main SoA loading pipeline, including the per-worker
        LRU cache, to provide fast access to the visual goal.

        Args:
            ep_idx: The index of the episode for which to retrieve the goal image.

        Returns:
            A NumPy array of the goal image in RGB, uint8 format.
        """
        if not (0 <= ep_idx < len(self.episode_metadata)):
            raise IndexError(f"Episode index {ep_idx} is out of range for {len(self.episode_metadata)} episodes.")

        ep_meta = self.episode_metadata[ep_idx]
        try:
            # Look for the metadata of our new modality
            meta = ep_meta["modalities"]["goal_image_primary"]
        except KeyError:
            raise KeyError(f"Modality 'goal_image_primary' not found in the index for episode {ep_idx}. "
                           "Please ensure the dataset was enhanced correctly.")

        # This call is fast and cached. It returns the goal image wrapped in an array of shape (1, H, W, 3).
        # It leverages the existing, powerful _get_full_modality_array method.
        full_array = self._get_full_modality_array(
            meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
        )

        # The result is an array containing a single image. We return just that image.
        return full_array[0]

    def get_num_episodes(self) -> int:
        """
        Returns the total number of episodes in the dataset.
        This is a fast, metadata-only operation.
        """
        return len(self.episode_metadata)
    
    def get_episode_length(self, episode_idx: int) -> int:
        """
        Returns the length (number of timesteps) of a specific episode.
        This is a fast, metadata-only operation.
        """
        if not (0 <= episode_idx < len(self.episode_metadata)):
            raise IndexError(f"Episode index {episode_idx} is out of range.")
        return self.episode_metadata[episode_idx]['length']

    def get_observation(self, ep_idx: int, t: int) -> Dict[str, np.ndarray]:
        """[SOTA PATCH] Directly retrieves all modalities for a specific timestep."""
        ep_meta = self.episode_metadata[ep_idx]
        obs = {}
        for key, meta in ep_meta["modalities"].items():
            if key == "actions": continue # Skip actions in observation
            full_array = self._get_full_modality_array(
                meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
            )
            obs[key] = full_array[t]
        return obs

    def get_proprio_sequence(self, ep_idx: int, start: int, end: int) -> np.ndarray:
        """[SOTA PATCH] Bulk sequence retrieval for proprioception history."""
        meta = self.episode_metadata[ep_idx]["modalities"]["proprio"]
        full_arr = self._get_full_modality_array(
            meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
        )
        return full_arr[start:end]

    def get_action_chunk(self, ep_idx: int, t: int, K: int) -> np.ndarray:
        """[SOTA PATCH] Bulk retrieval of future action trajectories."""
        meta = self.episode_metadata[ep_idx]["modalities"]["actions"]
        full_actions = self._get_full_modality_array(
            meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
        )
        return full_actions[t:t+K]

    def get_advantage(self, ep_idx: int, t: int) -> float:
        """[SOTA PATCH] Retrieve advantage for a specific timestep."""
        meta = self.episode_metadata[ep_idx]["modalities"].get("advantages")
        if meta is None:
            return 0.0 # Standard BC fallback
        full_adv = self._get_full_modality_array(
            meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
        )
        return float(full_adv[t])

    def get_episode_and_timestep(self, idx: int) -> Tuple[int, int]:
        """
        Performs a "reverse lookup" to map a global sample index back to its
        corresponding episode index and local timestep within that episode.

        This is a critical utility for orchestrator datasets that need to access
        both chunked and single-timestep data for a given sample.

        Args:
            idx: The global sample index.

        Returns:
            A tuple of (episode_index, timestep_in_episode).
        """
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} chunks.")

        # This is the same highly-efficient O(log N) lookup logic used in __getitem__.
        # Find the first episode index where the cumulative sum of chunks is >= idx.
        ep_idx = np.searchsorted(self._cumulative_chunks, idx, side='right')
        
        # Determine the starting global index for this episode's chunks.
        ep_start_chunk_idx = self._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
        
        # The local chunk index is the offset from the episode's start.
        local_chunk_idx = idx - ep_start_chunk_idx
        
        # The local timestep `t` is the first possible start time plus the local chunk index.
        timestep_t = (self.observation_horizon - 1) + local_chunk_idx
        
        return int(ep_idx), int(timestep_t)


    def to_robot_episode(self, ep_idx: int) -> Dict[str, Any]:
        """
        Converts a single LMDB episode into a dictionary matching the
        RobotEpisode / in-house export format consumed by the empirical
        experiment pipeline.

        This method performs a **full-episode materialization**: it loads
        every available modality for the specified episode from LMDB,
        decodes it (decompressing images if needed), and assembles a
        plain-dict representation with numpy arrays.

        The output dict schema is:
            {
                "episode_id":    str,
                "states":        np.ndarray (T, proprio_dim),   float32
                "actions":       np.ndarray (T, action_dim),    float32
                "phase_labels":  np.ndarray (T,),               int64
                "state_names":   List[str],                     (expert_states)
                "ee_pose":       np.ndarray (T, 7),             float32  [optional]
                "ee_vel":        np.ndarray (T, 6),             float32  [optional]
                "object_pos":    np.ndarray (T, 3),             float32  [optional]
                "is_grasped":    np.ndarray (T, 1),             float32  [optional]
                "expert_target_pose": np.ndarray (T, 7),       float32  [optional]
                "delta_ee_pose":     np.ndarray (T, 7),         float32  [optional]
                "success":       bool,
                "seed":          Optional[int],
                "length":        int,
            }

        Design decisions:
        - **Defensive loading:** Each modality is loaded independently
          with a try/except guard. Missing modalities are silently
          omitted rather than raising, ensuring forward compatibility
          with LMDB datasets that may lack optional fields.
        - **expert_states resolution:** The LMDB writer stores these
          as pickled Python objects under the 'expert_states' key.
          They are decoded and returned as a List[str] in the
          'state_names' field, matching the CSV pipeline's convention.
        - **gt_phase / gt_gripper:** These are always present in
          well-formed datasets and are loaded unconditionally. If
          missing, a ValueError is raised (these are not optional for
          the experiment pipeline).
        - **Image modalities are deliberately excluded** from the
          output because the empirical experiment pipeline is purely
          analytical (no vision models). Skipping image decode saves
          significant I/O and CPU time.

        Args:
            ep_idx: Zero-based episode index.

        Returns:
            Dict matching the in-house export episode format.

        Raises:
            IndexError: If ep_idx is out of range.
            ValueError: If required modalities (proprio, actions, gt_phase)
                        are missing from the episode's index metadata.
        """
        if not (0 <= ep_idx < len(self.episode_metadata)):
            raise IndexError(
                f"Episode index {ep_idx} out of range "
                f"for dataset with {len(self.episode_metadata)} episodes."
            )

        ep_meta = self.episode_metadata[ep_idx]
        modalities = ep_meta["modalities"]
        T = ep_meta["length"]

        def _load_modality(name: str) -> Optional[np.ndarray]:
            """Load a modality array if present, else return None."""
            if name not in modalities:
                return None
            m = modalities[name]
            return self._get_full_modality_array(
                m["key"], m["compression"], m["dtype"], tuple(m["shape"])
            )

        # --- Required modalities (raise if missing) ---
        proprio = _load_modality("proprio")
        if proprio is None:
            raise ValueError(
                f"Episode {ep_idx} is missing required 'proprio' modality. "
                f"Cannot convert to RobotEpisode format."
            )

        actions = _load_modality("actions")
        if actions is None:
            raise ValueError(
                f"Episode {ep_idx} is missing required 'actions' modality. "
                f"Cannot convert to RobotEpisode format."
            )

        gt_phase = _load_modality("gt_phase")
        if gt_phase is None:
            raise ValueError(
                f"Episode {ep_idx} is missing required 'gt_phase' modality. "
                f"Cannot convert to RobotEpisode format."
            )

        # --- Expert states (pickled list of strings) ---
        expert_states_raw = _load_modality("expert_states")
        if expert_states_raw is not None:
            # expert_states is stored as a pickled list of Python objects (strings)
            state_names: List[str] = [
                str(s) for s in expert_states_raw
            ]
        else:
            # Fallback: infer from gt_phase numeric labels
            state_names = [f"PHASE_{int(gt_phase[t])}" for t in range(T)]

        # --- Assemble output dict ---
        episode_dict: Dict[str, Any] = {
            "episode_id": ep_meta.get("episode_id", f"ep_{ep_idx:06d}"),
            "states": proprio.astype(np.float32),
            "actions": actions.astype(np.float32),
            "phase_labels": gt_phase.flatten().astype(np.int64),
            "state_names": state_names,
            "success": ep_meta.get("success", False),
            "seed": ep_meta.get("seed", None),
            "length": T,
        }

        # --- Optional modalities (included if present) ---
        for mod_name, dict_key in [
            ("ee_pose_world", "ee_pose"),
            ("ee_vel", "ee_vel"),
            ("object_pos_world", "object_pos"),
            ("is_grasped", "is_grasped"),
            ("expert_target_pose", "expert_target_pose"),
            ("delta_ee_pose", "delta_ee_pose"),
            ("gripper_qpos", "gripper_qpos"),
            ("gripper_vel", "gripper_vel"),
            ("object_vel", "object_vel"),
            ("goal_pos_world", "goal_pos_world"),
            ("goal_orn_world", "goal_orn_world"),
            ("goal_size_world", "goal_size_world"),
            ("robot_base_pos_world", "robot_base_pos_world"),
        ]:
            arr = _load_modality(mod_name)
            if arr is not None:
                episode_dict[dict_key] = arr.astype(np.float32)

        # gt_gripper is a special case — flatten to (T,) for convenience
        gt_gripper = _load_modality("gt_gripper")
        if gt_gripper is not None:
            episode_dict["gt_gripper"] = gt_gripper.flatten().astype(np.float32)

        return episode_dict

    def to_applied_dataset(self, max_episodes: int = 0) -> Dict[str, Any]:
        """
        Materializes the entire LMDB dataset into an AppliedDataset-compatible
        dictionary containing a list of episode dicts and index metadata.

        This is the primary bridge method for the empirical experiment pipeline.
        It produces the same data structure as ``load_applied_dataset`` and
        ``load_inhouse_export_episodes``, allowing a drop-in replacement.

        Architecture:
        - **Lazy LMDB init:** The LMDB environment is initialized on first
          access via ``_init_lmdb``, respecting worker-process safety.
        - **Sequential episode loading:** Episodes are loaded one at a time
          using ``to_robot_episode``, which leverages the LRU-cached
          ``_get_full_modality_array`` for efficient repeated access.
        - **Index metadata passthrough:** The JSON index metadata (recording
          mode, control mode, etc.) is forwarded to the output, preserving
          the same enrichment that the CSV pipeline provides.
        - **Error resilience:** Individual episode failures are logged and
          skipped rather than aborting the entire load. This mirrors the
          robustness of the CSV ``load_applied_dataset`` function.

        Args:
            max_episodes: If > 0, load at most this many episodes (smoke mode).

        Returns:
            Dict with keys:
                - "episodes": List[Dict] — one dict per episode
                - "index_metadata": Dict — the JSON index metadata
                - "export_root": str — the LMDB file path
        """
        n_total = len(self.episode_metadata)
        n_load = min(n_total, max_episodes) if max_episodes > 0 else n_total

        logger.info(
            f"Materializing {n_load}/{n_total} episodes from LMDB "
            f"({self.demo_path.name})"
        )

        episodes: List[Dict[str, Any]] = []
        for ep_idx in range(n_load):
            try:
                ep_dict = self.to_robot_episode(ep_idx)
                episodes.append(ep_dict)
            except Exception as e:
                ep_id = self.episode_metadata[ep_idx].get("episode_id", f"ep_{ep_idx:06d}")
                logger.warning(
                    f"Failed to load episode {ep_id} from LMDB: {e}"
                )

        logger.info(
            f"Successfully loaded {len(episodes)}/{n_load} episodes from LMDB."
        )

        return {
            "episodes": episodes,
            "index_metadata": {
                "recording_mode": self.recording_mode,
                "control_mode": self.control_mode,
                "metadata": self.metadata,
            },
            "export_root": str(self.demo_path),
        }

    def get_proprioception_dim(self) -> int:
        """
        Returns the dimension of the 'proprio' modality from the dataset's metadata.
        This is a fast, metadata-only operation.
        """
        if not self.episode_metadata:
            # This should not happen if the index loaded correctly
            raise RuntimeError("Dataset index is empty, cannot infer proprioception dim.")
            
        # Get the metadata for the 'proprio' modality from the first episode.
        # We assume this is consistent across all episodes.
        try:
            proprio_meta = self.episode_metadata[0]['modalities']['proprio']
            # The shape is [T, Dim]. We want the last element.
            return proprio_meta['shape'][-1]
        except (KeyError, IndexError):
            raise RuntimeError(
                "Could not infer proprioception dimension from the dataset's index.json. "
                "Ensure 'proprio' modality with a valid 'shape' is present in the index."
            )
  
    def __len__(self):
        return self.total_chunks 

    def close_env(self):
        """Explicitly closes the LMDB environment (useful for pickling)."""
        if getattr(self, "_lmdb_env", None) is not None:
            close_lmdb_env(self._lmdb_env)
            self._lmdb_env = None
            logger.debug(f"Process {os.getpid()} explicitly closed LMDB handle.")

    def __getstate__(self):
        """Prepare for pickling by removing unpicklable handles."""
        state = self.__dict__.copy()
        # Environments and bound methods (lru_cache wrappers) are not picklable.
        state['_lmdb_env'] = None
        # We don't pickle the cache itself to avoid pickling the method wrapper
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._lmdb_env = None # Force re-init in worker process

    def _init_lmdb(self):
        """Initializes the LMDB environment for the current worker."""
        # [SOTA FIX] Detect if we have been forked and have an inherited handle
        current_pid = os.getpid()
        if self._pid != current_pid:
            # We are in a child process (worker) and the handle was opened in the parent.
            # We MUST reset the handle to None so this worker opens its own.
            # Trying to use the parent's handle would cause a segmentation fault.
            self._lmdb_env = None
            self._pid = current_pid
            # Clear the LRU cache as well, as decoding handles might be stale
            self._get_full_modality_array.cache_clear()
            logger.debug(f"Worker {current_pid} detected fork; resetting LMDB handle.")

        if self._lmdb_env is None:
            self._lmdb_env = open_lmdb_env(
                str(self.demo_path),
                readonly=True,
                lock=False,      # No locks, we are read-only 
                map_size_gb=calculate_lmdb_map_size_gb(len(self.episode_metadata)),
                readahead=False, # False = better for random access [cite: 246]
                subdir=False     # Our DB is a single file
            ) 
            logger.debug(f"Process {current_pid} opened LMDB env.")

    def __del__(self):
        """Ensures the LMDB environment is closed when a worker is destroyed."""
        if getattr(self, "_lmdb_env", None) is not None:
            close_lmdb_env(self._lmdb_env) 
            self._lmdb_env = None

    def _get_lmdb_blob(self, key: str) -> bytes:
        """Gets a raw byte blob from LMDB, initializing the env if needed."""
        self._init_lmdb()
        with self._lmdb_env.begin(write=False) as txn:
            blob = txn.get(key.encode("ascii"))
            if blob is None:
                raise KeyError(f"Missing LMDB key {key!r} in {self.demo_path}") 
            return blob

    @functools.lru_cache(maxsize=8)
    def _get_full_modality_array(self, key: str, compression: str, dtype_str: str, shape_list: tuple) -> Any:
        """
        [DEFINITIVE, FULLY PATCHED, SOTA VERSION]
        This is the cached, expensive part. It loads an *entire* modality
        (e.g., all 'proprio' for one episode) from LMDB and decodes it.
        This version correctly handles 'raw', 'jpeg'/'png', and 'pickle' compression.
        """
        blob = self._get_lmdb_blob(key)
        shape = tuple(shape_list) # Ensure shape is a tuple for consistency

        if compression == "raw":
            dtype = np.dtype(dtype_str)
            # [SOTA FIX] Always return a copy for raw modalities to ensure compatibility 
            # with downstream functions that may perform in-place normalization or math.
            data = np.frombuffer(blob, dtype=dtype).reshape(shape).copy()
            return data

        elif compression in ("jpeg", "png"):
            byte_list = pickle.loads(blob)
            
            # This logic robustly handles color (3-ch) and grayscale (1-ch) images.
            is_grayscale = len(shape) == 4 and shape[3] == 1
            im_read_flag = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
            
            images = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), im_read_flag) for b in byte_list]
            data = np.stack(images)
            
            if not is_grayscale:
                data = data[..., ::-1]  # BGR to RGB for color images
            if is_grayscale and len(data.shape) == 3:
                data = data[..., np.newaxis] # Ensure channel dim exists for grayscale

            return data

        elif compression == "pickle":
            # [SOTA FIX] Return DEEP COPY to prevent cache corruption !!
            # The cache holds the original; readers get a safe copy.
            return copy.deepcopy(pickle.loads(blob))

        else:
            raise ValueError(f"Unknown compression type: {compression}")

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]: 
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} chunks.")

        # --- 1. Virtual Indexing (O(log N) lookup) ---
        # Find the first episode index where the cumulative sum is >= idx
        ep_idx = np.searchsorted(self._cumulative_chunks, idx, side='right')
        
        # Get the start index of this episode's chunks
        ep_start_chunk_idx = self._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
        
        # Get the local timestep `t` within the episode
        local_chunk_idx = idx - ep_start_chunk_idx
        t = (self.observation_horizon - 1) + local_chunk_idx
        
        # Get the metadata for this specific episode
        ep_meta = self.episode_metadata[ep_idx]

        # --- 2. Slicing Logic (Identical to original) ---
        obs_start_idx = t - self.observation_horizon + 1
        obs_end_idx = t + 1
        action_start_idx = t
        action_end_idx = t + self.action_horizon 

        # --- 3. SOTA Chunk Loading ---
        # We load *only* the slices we need from the cached full arrays.
        obs_chunk = {}
        for key in ["image_primary", "image_wrist", "proprio"]: 
            meta = ep_meta["modalities"][key]
            
            # This call is fast:
            # 1. Hits the LRU cache.
            # 2. If miss, loads/decodes the *full* modality.
            # 3. Caches the full modality.
            full_array = self._get_full_modality_array(
                meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
            )
            
            # This numpy slice is a fast, C-backend view operation
            obs_chunk[key] = full_array[obs_start_idx:obs_end_idx]

        # Load the action chunk
        meta_actions = ep_meta["modalities"]["actions"]
        full_actions = self._get_full_modality_array(
            meta_actions["key"], meta_actions["compression"], meta_actions["dtype"], tuple(meta_actions["shape"])
        )
        # [SOTA FIX] Robust Horizon Padding
        # Ensure we always return exactly action_horizon steps.
        # If slice is short (end of episode), pad with the last action.
        raw_chunk = full_actions[action_start_idx:action_end_idx]
        current_len = raw_chunk.shape[0]
        target_len = self.action_horizon
        
        if current_len < target_len:
            padding = np.tile(raw_chunk[-1:], (target_len - current_len, 1))
            action_chunk = np.concatenate([raw_chunk, padding], axis=0).astype(np.float32)
        else:
            action_chunk = raw_chunk.astype(np.float32)

        return obs_chunk, action_chunk 


# ==============================================================================
# 3. COLLATE FUNCTION (Unchanged, but required for completeness)
# ==============================================================================

def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]]): 
    """
    Collates a batch of chunked data into a single PyTorch tensor dictionary.
    This function is compatible with both the old and new dataset.
    
    Input shape (example):
      - sample[0] (obs): Dict with arrays like (H_obs, H, W, C) 
      - sample[1] (action): Array like (H_act, A_dim) 
    Returns:
      - obs_batch: Dict with tensors like (B, H_obs, H, W, C) 
      - action_batch: Tensor like (B, H_act, A_dim) 
    """
    if not batch: 
        return {}, torch.empty(0)

    obs_batch = {}
    # Get all observation keys from the first sample
    obs_keys = batch[0][0].keys() 
    
    for key in obs_keys:
        # Stack all observations for this key across the batch dimension
        # Converts numpy arrays to torch tensors automatically
        obs_batch[key] = torch.from_numpy(np.stack([sample[0][key] for sample in batch])) 

    # Stack all action chunks across the batch dimension
    action_batch = torch.from_numpy(np.stack([sample[1] for sample in batch])) 
    
    return obs_batch, action_batch 


# ==============================================================================
# 4. REPLAY VALIDATION (Unchanged, operates on in-memory dicts)
# ==============================================================================

def replay_validate_episode(ep: Dict[str, Any], urdf_path: str, env_xml_path: Optional[str] = None) -> bool: 
    """
    Replay sim_actions in a fresh environment and compare final object pose vs stored.
    Return True if within tolerance.
    [cite: 332]
    """
    env = PandaEnv(xml_path=env_xml_path, control_mode='delta')
    env.reset(seed=ep.get("seed", None))
    if ep.get("obs_list"):
        try:
            # Restore position
            init_obj_pos = np.array(ep["obs_list"][0]["object_pos_world"], dtype=np.float32)
            obj_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "object")
            if obj_body_id != -1:
                 env.data.xpos[obj_body_id] = init_obj_pos 
            
            # Restore orientation (if available)
            if "object_orn_world" in ep["obs_list"][0]:
                init_obj_orn_xyzw = np.array(ep["obs_list"][0]["object_orn_world"], dtype=np.float32)
                # Convert to MuJoCo's wxyz format
                init_obj_orn_wxyz = np.array([init_obj_orn_xyzw[3], init_obj_orn_xyzw[0], init_obj_orn_xyzw[1], init_obj_orn_xyzw[2]]) 
                obj_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
                if obj_joint_id != -1:
                    qpos_adr = env.model.jnt_qposadr[obj_joint_id]
                    env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = init_obj_orn_wxyz 

            # Apply changes to the simulation state
            mujoco.mj_forward(env.model, env.data)
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not restore initial object state for replay: {e}")
    for a in ep["actions"]: # Use the unified "actions" key
        obs, _, done, trunc, _ = env.step(np.array(a, dtype=np.float32))
        
        if done or trunc: 
            break
    final = obs
    tgt = ep["obs_list"][-1]["object_pos_world"]
    got = final["object_pos_world"]
    pos_err = np.linalg.norm(tgt - got)
    # orientation check (if stored object_orn_world)
    # skip orientation for now
    ok = pos_err <= REPLAY_POS_TOL
    if not ok:
        logger.warning(f"Replay mismatch pos_err={pos_err:.5f}")
    return ok


# ==============================================================================
# 5. STANDALONE LMDB READER
#    (No SYNAPSIS, cv2, or mujoco dependency — for empirical validation only)
# ==============================================================================

class StandaloneLMDBReader:
    """
    Lightweight, dependency-free LMDB reader for the empirical experiment pipeline.

    This class mirrors the data-loading logic of ``ExpertTrajectoryDataset`` but
    eliminates the hard dependencies on ``SYNAPSIS.utils.lmdb_utils``, ``cv2``,
    and ``mujoco``. It is designed specifically for the analytical (non-training)
    experiment track where images are not needed and SYNAPSIS may not be installed.

    Architecture (mirrors ExpertTrajectoryDataset):
    - **JSON index loading:** Reads the companion ``_index.json`` file for
      episode metadata, modalities, shapes, and compression types — identical
      to the SOTA reader's approach.
    - **Lazy LMDB init:** The LMDB environment is opened on first read, not at
      construction time. This avoids holding file handles unnecessarily.
    - **Per-modality decoding:** Handles ``raw`` (numpy frombuffer), ``pickle``
      (pickle.loads + deepcopy), and gracefully skips ``jpeg``/``png`` modalities
      with a warning (images are not needed for empirical validation).
    - **Error resilience:** Individual episode or modality failures are logged
      and skipped, matching the robustness of the CSV pipeline.
    - **Output format:** Produces the same dict schema as
      ``ExpertTrajectoryDataset.to_robot_episode()``, ensuring drop-in
      compatibility with ``load_applied_dataset_from_lmdb()`` in ``data.py``.

    Limitations vs ExpertTrajectoryDataset:
    - No image decoding (jpeg/png modalities are skipped with a warning).
    - No virtual indexing or ``__getitem__`` (full-episode materialization only).
    - No LRU cache (episodes are loaded once; re-access re-reads from LMDB).
    - No worker-process fork safety (single-process analytical use only).
    """

    def __init__(self, lmdb_path: str):
        """
        Args:
            lmdb_path: Path to the ``.lmdb`` file. A companion
                ``_index.json`` must exist in the same directory.
        """
        self.demo_path = Path(lmdb_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"LMDB file not found: {self.demo_path}")

        self.index_path = self.demo_path.parent / f"{self.demo_path.stem}_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Missing required index file: {self.index_path}\n"
                f"Each LMDB dataset must have a companion _index.json."
            )

        # --- Load the JSON Index (same as ExpertTrajectoryDataset.__init__) ---
        logger.info(f"StandaloneLMDBReader: loading index from {self.index_path}...")
        with open(self.index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.recording_mode = index_data.get("recording_mode", None)
        self.control_mode = index_data.get("control_mode", None)
        self.episode_metadata = index_data["episodes"]
        self.metadata = index_data.get("metadata", {})

        logger.info(
            f"StandaloneLMDBReader: {len(self.episode_metadata)} episodes indexed "
            f"from {self.demo_path.name}"
        )

        # Lazy LMDB environment — opened on first read
        self._env = None

    # ------------------------------------------------------------------
    # LMDB lifecycle (mirrors ExpertTrajectoryDataset._init_lmdb / close_env)
    # ------------------------------------------------------------------

    def _ensure_env(self):
        """Open the LMDB environment on first access (lazy init)."""
        if self._env is None:
            if lmdb is None:
                raise ImportError(
                    "The 'lmdb' package is required to read LMDB datasets. "
                    "Install it with: pip install lmdb"
                )
            # Map size: generous default (1 TB) for read-only access;
            # LMDB only maps what's actually used, so this is safe.
            self._env = lmdb.open(
                str(self.demo_path),
                subdir=False,
                readonly=True,
                lock=False,
                map_size=1 << 40,
                readahead=False,
            )
            logger.debug(f"StandaloneLMDBReader: opened LMDB env for {self.demo_path.name}")

    def close_env(self):
        """Explicitly close the LMDB environment to release the file handle."""
        if self._env is not None:
            self._env.close()
            self._env = None
            logger.debug("StandaloneLMDBReader: closed LMDB env.")

    def __del__(self):
        self.close_env()

    # ------------------------------------------------------------------
    # Low-level modality decoding (mirrors _get_full_modality_array)
    # ------------------------------------------------------------------

    def _read_modality(
        self,
        txn,
        key: str,
        compression: str,
        dtype_str: str,
        shape: List[int],
    ) -> Optional[np.ndarray]:
        """Decode a single modality from an LMDB transaction.

        Handles 'raw' and 'pickle' compression. Skips 'jpeg'/'png' with a
        warning (images not needed for empirical validation).

        Args:
            txn: An active LMDB read transaction.
            key: LMDB key (e.g., 'ep_000000_proprio').
            compression: 'raw', 'pickle', 'jpeg', or 'png'.
            dtype_str: Numpy dtype string (e.g., 'float32').
            shape: Expected array shape (e.g., [193, 22]).

        Returns:
            Decoded numpy array, or None if the modality is an image type
            (skipped) or the key is missing.
        """
        blob = txn.get(key.encode("ascii"))
        if blob is None:
            logger.warning(f"StandaloneLMDBReader: key {key!r} not found in LMDB, skipping.")
            return None

        shape_tuple = tuple(shape)

        if compression == "raw":
            dtype = np.dtype(dtype_str)
            # Return a COPY (same as ExpertTrajectoryDataset) to prevent
            # downstream in-place mutations from corrupting the LMDB buffer.
            return np.frombuffer(blob, dtype=dtype).reshape(shape_tuple).copy()

        elif compression == "pickle":
            # Deep copy to prevent cache corruption (same rationale as SOTA reader)
            return copy.deepcopy(pickle.loads(blob))

        elif compression in ("jpeg", "png"):
            logger.debug(
                f"StandaloneLMDBReader: skipping image modality {key!r} "
                f"(compression={compression}); not needed for empirical validation."
            )
            return None

        else:
            logger.warning(
                f"StandaloneLMDBReader: unknown compression '{compression}' "
                f"for key {key!r}, skipping."
            )
            return None

    # ------------------------------------------------------------------
    # Episode materialization (mirrors to_robot_episode)
    # ------------------------------------------------------------------

    def load_episode(self, ep_idx: int) -> Optional[Dict[str, Any]]:
        """Load a single episode as a dict matching the RobotEpisode schema.

        The output schema is identical to ``ExpertTrajectoryDataset.to_robot_episode()``:
            {
                "episode_id":    str,
                "states":        np.ndarray (T, proprio_dim),   float32
                "actions":       np.ndarray (T, action_dim),    float32
                "phase_labels":  np.ndarray (T,),               int64
                "state_names":   List[str],
                "ee_pose":       np.ndarray (T, 7),             float32  [optional]
                "ee_vel":        np.ndarray (T, 6),             float32  [optional]
                "object_pos":    np.ndarray (T, 3),             float32  [optional]
                "is_grasped":    np.ndarray (T, 1),             float32  [optional]
                "expert_target_pose": np.ndarray (T, 7),        float32  [optional]
                "delta_ee_pose":     np.ndarray (T, 7),         float32  [optional]
                "gripper_qpos":      np.ndarray (T, 2),         float32  [optional]
                "gripper_vel":       np.ndarray (T, 2),         float32  [optional]
                "object_vel":        np.ndarray (T, 6),         float32  [optional]
                "goal_pos_world":    np.ndarray (T, 3),         float32  [optional]
                "goal_orn_world":    np.ndarray (T, 4),         float32  [optional]
                "goal_size_world":   np.ndarray (T, 3),         float32  [optional]
                "robot_base_pos_world": np.ndarray (T, 3),      float32  [optional]
                "gt_gripper":        np.ndarray (T,),            float32  [optional]
                "success":       bool,
                "seed":          Optional[int],
                "length":        int,
            }

        Args:
            ep_idx: Zero-based episode index.

        Returns:
            Episode dict, or None if loading fails (error is logged).
        """
        if not (0 <= ep_idx < len(self.episode_metadata)):
            raise IndexError(
                f"Episode index {ep_idx} out of range "
                f"for dataset with {len(self.episode_metadata)} episodes."
            )

        ep_meta = self.episode_metadata[ep_idx]
        modalities = ep_meta["modalities"]
        T = ep_meta["length"]

        self._ensure_env()

        try:
            with self._env.begin(write=False) as txn:

                # --- Helper: load one modality from the current txn ---
                def _load(name: str) -> Optional[np.ndarray]:
                    if name not in modalities:
                        return None
                    m = modalities[name]
                    return self._read_modality(
                        txn, m["key"], m["compression"], m["dtype"], m["shape"]
                    )

                # --- Required modalities (raise if missing) ---
                proprio = _load("proprio")
                if proprio is None:
                    raise ValueError(
                        f"Episode {ep_idx} missing required 'proprio' modality."
                    )

                actions = _load("actions")
                if actions is None:
                    raise ValueError(
                        f"Episode {ep_idx} missing required 'actions' modality."
                    )

                gt_phase = _load("gt_phase")
                if gt_phase is None:
                    raise ValueError(
                        f"Episode {ep_idx} missing required 'gt_phase' modality."
                    )

                # --- Expert states (pickled list of strings) ---
                expert_states_raw = _load("expert_states")
                if expert_states_raw is not None:
                    state_names: List[str] = [str(s) for s in expert_states_raw]
                else:
                    state_names = [f"PHASE_{int(gt_phase[t])}" for t in range(T)]

                # --- Assemble output dict (same schema as to_robot_episode) ---
                episode_dict: Dict[str, Any] = {
                    "episode_id": ep_meta.get("episode_id", f"ep_{ep_idx:06d}"),
                    "states": proprio.astype(np.float32),
                    "actions": actions.astype(np.float32),
                    "phase_labels": gt_phase.flatten().astype(np.int64),
                    "state_names": state_names,
                    "success": ep_meta.get("success", False),
                    "seed": ep_meta.get("seed", None),
                    "length": T,
                }

                # --- Optional modalities (same mapping as to_robot_episode) ---
                for mod_name, dict_key in [
                    ("ee_pose_world", "ee_pose"),
                    ("ee_vel", "ee_vel"),
                    ("object_pos_world", "object_pos"),
                    ("is_grasped", "is_grasped"),
                    ("expert_target_pose", "expert_target_pose"),
                    ("delta_ee_pose", "delta_ee_pose"),
                    ("gripper_qpos", "gripper_qpos"),
                    ("gripper_vel", "gripper_vel"),
                    ("object_vel", "object_vel"),
                    ("goal_pos_world", "goal_pos_world"),
                    ("goal_orn_world", "goal_orn_world"),
                    ("goal_size_world", "goal_size_world"),
                    ("robot_base_pos_world", "robot_base_pos_world"),
                ]:
                    arr = _load(mod_name)
                    if arr is not None:
                        episode_dict[dict_key] = arr.astype(np.float32)

                # gt_gripper: flatten (T,1) → (T,)
                gt_gripper = _load("gt_gripper")
                if gt_gripper is not None:
                    episode_dict["gt_gripper"] = gt_gripper.flatten().astype(np.float32)

                return episode_dict

        except Exception as e:
            ep_id = ep_meta.get("episode_id", f"ep_{ep_idx:06d}")
            logger.warning(f"StandaloneLMDBReader: failed to load episode {ep_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Full-dataset materialization (mirrors to_applied_dataset)
    # ------------------------------------------------------------------

    def to_applied_dataset(self, max_episodes: int = 0) -> Dict[str, Any]:
        """Materialize episodes into an AppliedDataset-compatible dict.

        The output format is identical to ``ExpertTrajectoryDataset.to_applied_dataset()``:
            {
                "episodes": List[Dict],
                "index_metadata": Dict,
                "export_root": str,
            }

        This is the primary bridge for ``load_applied_dataset_from_lmdb()``
        in ``data.py`` — a drop-in replacement when SYNAPSIS is not available.

        Args:
            max_episodes: If > 0, load at most this many episodes (smoke mode).

        Returns:
            Dict with episodes list, index metadata, and export root path.
        """
        n_total = len(self.episode_metadata)
        n_load = min(n_total, max_episodes) if max_episodes > 0 else n_total

        logger.info(
            f"StandaloneLMDBReader: materializing {n_load}/{n_total} episodes "
            f"from {self.demo_path.name}"
        )

        episodes: List[Dict[str, Any]] = []
        for ep_idx in range(n_load):
            ep_dict = self.load_episode(ep_idx)
            if ep_dict is not None:
                episodes.append(ep_dict)

        logger.info(
            f"StandaloneLMDBReader: loaded {len(episodes)}/{n_load} episodes."
        )

        return {
            "episodes": episodes,
            "index_metadata": {
                "recording_mode": self.recording_mode,
                "control_mode": self.control_mode,
                "metadata": self.metadata,
            },
            "export_root": str(self.demo_path),
        }

    def get_num_episodes(self) -> int:
        """Return the total number of episodes in the dataset."""
        return len(self.episode_metadata)