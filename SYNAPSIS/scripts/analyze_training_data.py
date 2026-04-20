# FILE: scripts/analyze_training_data.py
# (PhD-Rigor, multi-dimensional Dataset Audit & Quality Assurance Engine)

import argparse
import logging
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm
from scipy import stats

# Ensure the project root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SYNAPSIS.datasets.expert_dataset import ExpertTrajectoryDataset

# Configure UTF-8 for Windows CLI robustness
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [DATA_AUDIT] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "data_quality_report.log", mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger("Audit")

# Phase map for Panda Env
PHASE_MAP = {
    0: "0_Approach",
    1: "1_Grasp",
    2: "2_Transport",
    3: "3_Place",
    4: "4_Retract"
}

class DataAuditor:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        log.info(f"🚀 Initializing Audit for: {self.dataset_path.name}")
        self.ds = ExpertTrajectoryDataset(
            demo_path=str(self.dataset_path),
            observation_horizon=1,
            action_horizon=1
        )
        self.num_episodes = self.ds.get_num_episodes()
        self.report = {
            "metadata": {
                "timestamp": time.ctime(),
                "dataset_name": self.dataset_path.name,
                "num_episodes": self.num_episodes
            },
            "checks": {},
            "warnings": []
        }

    def run_full_audit(self, exhaustive_modality_check=True):
        """Performs the complete PhD-grade suite of tests."""
        log.info("-" * 60)
        log.info(f"Audit Scope: {self.num_episodes} Episodes | {len(self.ds)} Transitions")
        log.info("-" * 60)

        # 1. Macro Statistics
        self._audit_macro_stats()
        
        # 2. Numerical Sanity & Integrity
        self._audit_numerical_integrity(exhaustive_modality_check)
        
        # 3. Phase Balance & Semantic Alignment
        self._audit_phase_balance()
        
        # 4. Temporal Consistency (Dynamics)
        self._audit_dynamics()
        
        # 5. RL Weighting & Advantage Distribution
        self._audit_advantages()

        self._generate_summary_report()

    def _audit_macro_stats(self):
        log.info("🔍 [1/5] Analyzing Macro Statistics...")
        lengths = []
        successes = 0
        
        for ep_idx in range(self.num_episodes):
            meta = self.ds.episode_metadata[ep_idx]
            lengths.append(meta['length'])
            if meta.get('success', False):
                successes += 1
        
        lengths = np.array(lengths)
        self.report["checks"]["macro"] = {
            "success_rate": successes / self.num_episodes,
            "horizon": {
                "avg": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "median": float(np.median(lengths))
            }
        }
        log.info(f"✅ Success Rate: {self.report['checks']['macro']['success_rate']*100:.2f}%")
        log.info(f"✅ Horizon: Avg={lengths.mean():.1f}, Min={lengths.min()}, Max={lengths.max()}")

    def _audit_numerical_integrity(self, exhaustive):
        log.info("🔍 [2/5] Performing Numerical Integrity Check (NaN/Inf/Flat)...")
        nan_indices = []
        inf_indices = []
        dead_channels = {} # Modality -> list of indices

        # Modalities to check
        test_keys = ['actions', 'proprio', 'ee_pose_world', 'object_pos_world']
        
        for ep_idx in tqdm(range(self.num_episodes), desc="Integrity Scan"):
            ep_meta = self.ds.episode_metadata[ep_idx]
            for key in test_keys:
                if key not in ep_meta['modalities']: continue
                
                meta = ep_meta['modalities'][key]
                data = self.ds._get_full_modality_array(meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape']))
                
                # NaN/Inf
                if np.isnan(data).any(): nan_indices.append((ep_idx, key))
                if np.isinf(data).any(): inf_indices.append((ep_idx, key))
                
                # Zero Variance (Dead Channels)
                if exhaustive:
                    var = np.var(data, axis=0)
                    dead = np.where(var < 1e-12)[0]
                    if len(dead) > 0:
                        if key not in dead_channels: dead_channels[key] = set()
                        dead_channels[key].update(dead.tolist())

        if nan_indices:
            log.error(f"❌ NaNs Detected in {len(nan_indices)} episodes!")
            self.report["warnings"].append(f"NaNs detected in keys: {list(set([x[1] for x in nan_indices]))}")
        
        if inf_indices:
            log.error(f"❌ Infinities Detected in {len(inf_indices)} episodes!")
            self.report["warnings"].append(f"Infinities detected")

        # Report Dead Channels
        dead_final = {k: sorted(list(v)) for k, v in dead_channels.items()}
        self.report["checks"]["integrity"] = {
            "nan_count": len(nan_indices),
            "inf_count": len(inf_indices),
            "dead_channels": dead_final
        }
        
        if not nan_indices and not inf_indices:
            log.info("✅ Numerical hygiene verified. No NaNs or Infs found.")

    def _audit_phase_balance(self):
        log.info("🔍 [3/5] Evaluating Phase Distribution...")
        phase_counts = sorted(PHASE_MAP.keys())
        global_phase_stats = {PHASE_MAP[p]: 0 for p in phase_counts}
        
        for ep_idx in range(self.num_episodes):
            ep_meta = self.ds.episode_metadata[ep_idx]
            if 'gt_phase' not in ep_meta['modalities']: continue
            
            meta = ep_meta['modalities']['gt_phase']
            phases = self.ds._get_full_modality_array(meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape']))
            
            for p_val in phases:
                p_int = int(p_val.item()) if hasattr(p_val, 'item') else int(p_val)
                p_name = PHASE_MAP.get(p_int, "Unknown")
                global_phase_stats[p_name] += 1
        
        total_steps = sum(global_phase_stats.values())
        if total_steps > 0:
            phase_pct = {k: v/total_steps for k, v in global_phase_stats.items()}
            self.report["checks"]["phase_balance"] = phase_pct
            for k, v in phase_pct.items():
                log.info(f"📊 {k:15}: {v*100:5.1f}% representation")
        else:
            log.warning("⚠️ No phase data found in dataset.")

    def _audit_dynamics(self):
        log.info("🔍 [4/5] Inspecting Temporal Dynamics (Smoothness)...")
        max_velocities = []
        jitter_score = [] # Acceleration magnitude variation

        for ep_idx in range(min(self.num_episodes, 100)): # Dynamic check on subset for speed
            ep_meta = self.ds.episode_metadata[ep_idx]
            if 'ee_pose_world' not in ep_meta['modalities']: continue
            
            meta = ep_meta['modalities']['ee_pose_world']
            poses = self.ds._get_full_modality_array(meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape']))
            
            # Compute Diff (Velocity)
            pos = poses[:, :3]
            vel = np.diff(pos, axis=0)
            max_vel = np.max(np.linalg.norm(vel, axis=1))
            max_velocities.append(max_vel)
            
            # Compute Second Diff (Acceleration)
            accel = np.diff(vel, axis=0)
            jitter = np.std(np.linalg.norm(accel, axis=1))
            jitter_score.append(jitter)

        if max_velocities:
            self.report["checks"]["dynamics"] = {
                "max_velocity_observed": float(np.max(max_velocities)),
                "avg_jitter_score": float(np.mean(jitter_score))
            }
            log.info(f"✅ Max EE Velocity: {self.report['checks']['dynamics']['max_velocity_observed']:.4f} m/step")
            if self.report["checks"]["dynamics"]["max_velocity_observed"] > 0.5:
                log.warning("⚠️ High velocity detected! Possible physics glitch or teleportation.")

    def _audit_advantages(self):
        log.info("🔍 [5/5] Validating RL Advantages & Weights...")
        all_advs = []
        
        for ep_idx in range(self.num_episodes):
            ep_meta = self.ds.episode_metadata[ep_idx]
            if 'advantages' not in ep_meta['modalities']: continue
            
            meta = ep_meta['modalities']['advantages']
            adv = self.ds._get_full_modality_array(meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape']))
            all_advs.append(adv)
            
        if all_advs:
            flat_adv = np.concatenate(all_advs)
            self.report["checks"]["advantages"] = {
                "mean": float(np.mean(flat_adv)),
                "std": float(np.std(flat_adv)),
                "min": float(np.min(flat_adv)),
                "max": float(np.max(flat_adv)),
                "skew": float(stats.skew(flat_adv))
            }
            log.info(f"✅ Advantages: Mean={flat_adv.mean():.4f}, Std={flat_adv.std():.4f}")
            log.info(f"✅ Distribution Skew: {self.report['checks']['advantages']['skew']:.4f}")
        else:
            log.warning("⚠️ No advantages found. Dataset might not have been processed by advantage_calculator.")

    def _generate_summary_report(self):
        report_path = ROOT / "DATA_QUALITY_REPORT.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=4)
        
        log.info("-" * 60)
        log.info("🏆 AUDIT COMPLETE")
        log.info(f"📁 Full report saved to: {report_path}")
        
        if self.report["warnings"]:
            log.warning(f"⚠️ TOTAL WARNINGS: {len(self.report['warnings'])}")
            for w in self.report["warnings"]:
                log.warning(f"  - {w}")
        else:
            log.info("🌟 DATASET IS SOTA-READY (No Critical Issues Detected)")
        log.info("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="PhD-Grade Data Quality Auditor")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training_set.lmdb")
    parser.add_argument("--exhaustive", action="store_true", help="Perform full dead-channel scan")
    args = parser.parse_args()

    try:
        auditor = DataAuditor(args.data_path)
        auditor.run_full_audit(exhaustive_modality_check=args.exhaustive)
    except Exception as e:
        log.error(f"🚨 Audit Failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
