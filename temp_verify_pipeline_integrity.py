#!/usr/bin/env python3
"""
PIPELINE INTEGRITY VERIFICATION
================================
Simulates the full execution flow and cross-checks:
  1. LMDB raw data (source of truth)
  2. Exported CSV (derived from LMDB via ExpertTrajectoryDataset)
  3. Mathematical correctness of cartesian_delta

This verifies the entire chain: Generation -> LMDB Write -> LMDB Read -> CSV Export
"""
import sys
import json
import pickle
import copy
import numpy as np
import pandas as pd
import lmdb
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LMDB_PATH = ROOT / "output_data/debug_lmdb/expert_debug_gen_2_episodes/expert_debug_gen_2_episodes.lmdb"
INDEX_PATH = ROOT / "output_data/debug_lmdb/expert_debug_gen_2_episodes/expert_debug_gen_2_episodes_index.json"
CSV_PATH = ROOT / "output_data/exports/episode_0_ep_000000/telemetry.csv"

def compute_delta_ee_pose(target_pose, current_pose, base_quat):
    """Exact replica of compute_delta_ee_pose from expert_dataset.py."""
    delta_pos_world = target_pose[:3] - current_pose[:3]
    R_base_world = R.from_quat(base_quat)
    delta_pos_base = R_base_world.inv().apply(delta_pos_world)
    R_current = R.from_quat(current_pose[3:])
    R_target = R.from_quat(target_pose[3:])
    R_delta_world = R_current.inv() * R_target
    R_base_inv = R_base_world.inv()
    R_delta_base = R_base_inv * R_delta_world
    delta_orn_xyzw = R_delta_base.as_quat().astype(np.float32)
    return np.concatenate([delta_pos_base, delta_orn_xyzw]).astype(np.float32)

def read_lmdb_raw(lmdb_path, key):
    """Read raw bytes from LMDB."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False, max_dbs=1)
    with env.begin(write=False) as txn:
        blob = txn.get(key.encode("ascii"))
    env.close()
    return blob

def main():
    print("="*70)
    print("PIPELINE INTEGRITY VERIFICATION")
    print("="*70)
    
    # Load index
    with open(INDEX_PATH, 'r') as f:
        index_data = json.load(f)
    
    ep_meta = index_data["episodes"][0]
    ep_id = ep_meta["episode_id"]
    ep_length = ep_meta["length"]
    modalities = ep_meta["modalities"]
    
    print(f"\nEpisode: {ep_id}")
    print(f"Length: {ep_length}")
    print(f"Recording mode: {index_data.get('recording_mode')}")
    print(f"Control mode: {index_data.get('control_mode')}")
    print(f"Modalities: {list(modalities.keys())}")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"\nCSV rows: {len(df)}, columns: {len(df.columns)}")
    
    errors = []
    checks_passed = 0
    
    # =========================================================================
    # CHECK 1: Episode length consistency
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 1] Episode Length Consistency")
    print("-"*70)
    if len(df) == ep_length:
        print(f"  ✅ CSV rows ({len(df)}) == LMDB length ({ep_length})")
        checks_passed += 1
    else:
        msg = f"CSV rows ({len(df)}) != LMDB length ({ep_length})"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 2: Actions integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 2] Actions Integrity (LMDB raw vs CSV)")
    print("-"*70)
    act_meta = modalities["actions"]
    act_blob = read_lmdb_raw(LMDB_PATH, act_meta["key"])
    act_lmdb = np.frombuffer(act_blob, dtype=np.dtype(act_meta["dtype"])).reshape(act_meta["shape"]).copy()
    
    # CSV actions columns
    act_cols = [f"actions_{i}" for i in range(8)]
    act_csv = df[act_cols].values
    
    if act_lmdb.shape == act_csv.shape:
        max_diff = np.max(np.abs(act_lmdb - act_csv))
        if max_diff < 1e-6:
            print(f"  ✅ Actions match (max diff: {max_diff:.10f})")
            checks_passed += 1
        else:
            msg = f"Actions mismatch (max diff: {max_diff})"
            print(f"  ❌ {msg}")
            errors.append(msg)
    else:
        msg = f"Actions shape mismatch: LMDB {act_lmdb.shape} vs CSV {act_csv.shape}"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 3: EE Pose integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 3] EE Pose World Integrity (LMDB raw vs CSV)")
    print("-"*70)
    ee_meta = modalities["ee_pose_world"]
    ee_blob = read_lmdb_raw(LMDB_PATH, ee_meta["key"])
    ee_lmdb = np.frombuffer(ee_blob, dtype=np.dtype(ee_meta["dtype"])).reshape(ee_meta["shape"]).copy()
    
    ee_cols = [f"ee_pose_world_{i}" for i in range(7)]
    ee_csv = df[ee_cols].values
    
    max_diff = np.max(np.abs(ee_lmdb - ee_csv))
    if max_diff < 1e-6:
        print(f"  ✅ EE Pose matches (max diff: {max_diff:.10f})")
        checks_passed += 1
    else:
        msg = f"EE Pose mismatch (max diff: {max_diff})"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 4: Expert Target Pose integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 4] Expert Target Pose Integrity (LMDB raw vs CSV)")
    print("-"*70)
    if "expert_target_pose" in modalities:
        tgt_meta = modalities["expert_target_pose"]
        tgt_blob = read_lmdb_raw(LMDB_PATH, tgt_meta["key"])
        tgt_lmdb = np.frombuffer(tgt_blob, dtype=np.dtype(tgt_meta["dtype"])).reshape(tgt_meta["shape"]).copy()
        
        tgt_cols = [f"expert_target_pose_{i}" for i in range(7)]
        tgt_csv = df[tgt_cols].values
        
        max_diff = np.max(np.abs(tgt_lmdb - tgt_csv))
        if max_diff < 1e-6:
            print(f"  ✅ Target Pose matches (max diff: {max_diff:.10f})")
            checks_passed += 1
        else:
            msg = f"Target Pose mismatch (max diff: {max_diff})"
            print(f"  ❌ {msg}")
            errors.append(msg)
    else:
        print("  ⚠️ expert_target_pose not in LMDB modalities")
    
    # =========================================================================
    # CHECK 5: Delta EE Pose integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 5] Delta EE Pose Integrity (LMDB raw vs CSV)")
    print("-"*70)
    if "delta_ee_pose" in modalities:
        delta_meta = modalities["delta_ee_pose"]
        delta_blob = read_lmdb_raw(LMDB_PATH, delta_meta["key"])
        delta_lmdb = np.frombuffer(delta_blob, dtype=np.dtype(delta_meta["dtype"])).reshape(delta_meta["shape"]).copy()
        
        delta_cols = [f"delta_ee_pose_{i}" for i in range(7)]
        delta_csv = df[delta_cols].values
        
        max_diff = np.max(np.abs(delta_lmdb - delta_csv))
        if max_diff < 1e-6:
            print(f"  ✅ Delta EE Pose matches (max diff: {max_diff:.10f})")
            checks_passed += 1
        else:
            msg = f"Delta EE Pose mismatch (max diff: {max_diff})"
            print(f"  ❌ {msg}")
            errors.append(msg)
    else:
        print("  ⚠️ delta_ee_pose not in LMDB modalities")
    
    # =========================================================================
    # CHECK 6: Cartesian Delta Mathematical Correctness
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 6] Cartesian Delta Mathematical Correctness")
    print("  Recomputing delta from target_pose, ee_pose, base_quat and comparing")
    print("-"*70)
    
    # Get base_quat from LMDB
    base_meta = modalities.get("robot_base_quat_world")
    if base_meta:
        # robot_base_quat_world might not be stored as a separate modality
        # Check if it's in proprio
        pass
    
    # Use CSV data for re-computation (already verified against LMDB)
    max_pos_diff = 0
    max_orn_diff = 0
    delta_mismatches = 0
    
    for t in range(ep_length):
        current_pose = ee_lmdb[t]
        target_pose = tgt_lmdb[t]
        
        # Get base_quat - check if stored separately or in proprio
        # For now, assume identity (robot base at origin)
        # We need to find the actual base_quat
        base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default assumption
        
        # Try to get from proprio or a dedicated field
        if "robot_base_quat_world" in modalities:
            bq_meta = modalities["robot_base_quat_world"]
            bq_blob = read_lmdb_raw(LMDB_PATH, bq_meta["key"])
            bq_lmdb = np.frombuffer(bq_blob, dtype=np.dtype(bq_meta["dtype"])).reshape(bq_meta["shape"]).copy()
            base_quat = bq_lmdb[t]
        
        expected_delta = compute_delta_ee_pose(target_pose, current_pose, base_quat)
        recorded_delta = delta_lmdb[t]
        
        pos_diff = np.linalg.norm(recorded_delta[:3] - expected_delta[:3])
        orn_diff = np.linalg.norm(recorded_delta[3:] - expected_delta[3:])
        
        max_pos_diff = max(max_pos_diff, pos_diff)
        max_orn_diff = max(max_orn_diff, orn_diff)
        
        if pos_diff > 1e-5 or orn_diff > 1e-5:
            delta_mismatches += 1
    
    print(f"  Max position diff: {max_pos_diff:.10f}")
    print(f"  Max orientation diff: {max_orn_diff:.10f}")
    print(f"  Mismatches: {delta_mismatches}/{ep_length}")
    
    if delta_mismatches == 0:
        print(f"  ✅ Cartesian delta mathematically correct")
        checks_passed += 1
    else:
        msg = f"Cartesian delta has {delta_mismatches} mismatches"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 7: Proprio integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 7] Proprio Integrity (LMDB raw vs CSV)")
    print("-"*70)
    proprio_meta = modalities["proprio"]
    proprio_blob = read_lmdb_raw(LMDB_PATH, proprio_meta["key"])
    proprio_lmdb = np.frombuffer(proprio_blob, dtype=np.dtype(proprio_meta["dtype"])).reshape(proprio_meta["shape"]).copy()
    
    proprio_cols = [f"proprio_{i}" for i in range(proprio_lmdb.shape[1])]
    proprio_csv = df[proprio_cols].values
    
    max_diff = np.max(np.abs(proprio_lmdb - proprio_csv))
    if max_diff < 1e-6:
        print(f"  ✅ Proprio matches (max diff: {max_diff:.10f})")
        checks_passed += 1
    else:
        msg = f"Proprio mismatch (max diff: {max_diff})"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 8: Object/Goal Pose integrity (LMDB vs CSV)
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 8] Object & Goal Pose Integrity (LMDB raw vs CSV)")
    print("-"*70)
    
    for mod_name in ["object_pos_world", "object_orn_world", "goal_pos_world", "goal_orn_world", "goal_size_world"]:
        if mod_name in modalities:
            m_meta = modalities[mod_name]
            m_blob = read_lmdb_raw(LMDB_PATH, m_meta["key"])
            m_lmdb = np.frombuffer(m_blob, dtype=np.dtype(m_meta["dtype"])).reshape(m_meta["shape"]).copy()
            
            m_cols = [f"{mod_name}_{i}" for i in range(m_lmdb.shape[1])]
            m_csv = df[m_cols].values
            
            max_diff = np.max(np.abs(m_lmdb - m_csv))
            if max_diff < 1e-6:
                print(f"  ✅ {mod_name} matches (max diff: {max_diff:.10f})")
                checks_passed += 1
            else:
                msg = f"{mod_name} mismatch (max diff: {max_diff})"
                print(f"  ❌ {msg}")
                errors.append(msg)
    
    # =========================================================================
    # CHECK 9: Actions == Delta EE Pose (first 7 dims) for cartesian_delta mode
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 9] Actions[0:7] == Delta EE Pose[0:7] (cartesian_delta mode)")
    print("-"*70)
    
    if index_data.get("recording_mode") == "cartesian_delta":
        # In cartesian_delta mode, the recorded action IS the delta_ee_pose + gripper
        # So actions[:, 0:7] should equal delta_ee_pose[:, 0:7]
        max_diff = np.max(np.abs(act_lmdb[:, :7] - delta_lmdb[:, :7]))
        if max_diff < 1e-6:
            print(f"  ✅ Actions[0:7] == Delta EE Pose[0:7] (max diff: {max_diff:.10f})")
            checks_passed += 1
        else:
            msg = f"Actions[0:7] != Delta EE Pose[0:7] (max diff: {max_diff})"
            print(f"  ❌ {msg}")
            errors.append(msg)
        
        # Actions[7] should be gripper
        print(f"  Actions[7] (gripper) range: [{act_lmdb[:, 7].min():.4f}, {act_lmdb[:, 7].max():.4f}]")
    else:
        print(f"  ⚠️ Skipping - recording_mode is {index_data.get('recording_mode')}")
    
    # =========================================================================
    # CHECK 10: Quaternion Normalization
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 10] Quaternion Normalization Check")
    print("-"*70)
    
    # EE pose quaternions
    ee_quat_norms = np.linalg.norm(ee_lmdb[:, 3:], axis=1)
    if np.all(np.abs(ee_quat_norms - 1.0) < 0.01):
        print(f"  ✅ EE Pose quaternions normalized (mean norm: {ee_quat_norms.mean():.6f})")
        checks_passed += 1
    else:
        msg = f"EE Pose quaternions NOT normalized (mean norm: {ee_quat_norms.mean():.6f})"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # Target pose quaternions
    tgt_quat_norms = np.linalg.norm(tgt_lmdb[:, 3:], axis=1)
    if np.all(np.abs(tgt_quat_norms - 1.0) < 0.01):
        print(f"  ✅ Target Pose quaternions normalized (mean norm: {tgt_quat_norms.mean():.6f})")
        checks_passed += 1
    else:
        msg = f"Target Pose quaternions NOT normalized (mean norm: {tgt_quat_norms.mean():.6f})"
        print(f"  ❌ {msg}")
        errors.append(msg)
    
    # =========================================================================
    # CHECK 11: Delta quaternion w-component sign consistency
    # =========================================================================
    print("\n" + "-"*70)
    print("[CHECK 11] Delta Quaternion w-component sign check")
    print("-"*70)
    
    # For small rotations, delta qw should be close to 1.0 (positive)
    # If qw is negative, it means the rotation is > 180 degrees, which is unusual
    # for smooth robot motions
    neg_qw_count = np.sum(delta_lmdb[:, 6] < 0)
    if neg_qw_count == 0:
        print(f"  ✅ All delta qw >= 0 (no sign flips)")
        checks_passed += 1
    else:
        print(f"  ⚠️ {neg_qw_count} timesteps with delta qw < 0 (may indicate large rotations)")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  Checks passed: {checks_passed}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\n  ❌ ERRORS DETECTED:")
        for i, e in enumerate(errors, 1):
            print(f"    {i}. {e}")
    else:
        print("\n  ✅ ALL CHECKS PASSED - DATA PIPELINE IS CORRECT")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
