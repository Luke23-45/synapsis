# FILE: scripts/export_episode.py
# (PhD-Rigor, Robust Episode Export Engine: CSV Telemetry + Multi-Camera Video)

import argparse
import logging
import os
import sys
import json
import cv2
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Ensure the project root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force UTF-8 for Windows consistency
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from SYNAPSIS.datasets.expert_dataset import ExpertTrajectoryDataset

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [EXPORT] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Export")

class EpisodeExporter:
    def __init__(self, dataset_path: str, output_root: str):
        self.dataset_path = Path(dataset_path)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        log.info(f"🚀 Initializing Exporter for: {self.dataset_path.name}")
        self.ds = ExpertTrajectoryDataset(
            demo_path=str(self.dataset_path),
            observation_horizon=1,
            action_horizon=1
        )
        self.num_episodes = self.ds.get_num_episodes()
        log.info(f"Dataset contains {self.num_episodes} episodes.")

    def export_episode(self, index: int, fps: int = 30):
        if index < 0 or index >= self.num_episodes:
            raise ValueError(f"Index {index} out of range [0, {self.num_episodes-1}]")

        ep_meta = self.ds.episode_metadata[index]
        ep_id = ep_meta['episode_id']
        length = ep_meta['length']
        export_dir = self.output_root / f"episode_{index}_{ep_id}"
        export_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"📦 Exporting Episode {index} (ID: {ep_id}) to: {export_dir}")
        log.info(f"Length: {length} steps")

        # 1. Classification of Modalities
        visual_modalities = []
        numeric_modalities = {} # name -> data_array

        for mod_name, meta in ep_meta['modalities'].items():
            data = self.ds._get_full_modality_array(
                meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape'])
            )
            
            if meta['compression'] == 'jpeg' or (len(meta['shape']) >= 3 and meta['shape'][-1] == 3):
                visual_modalities.append((mod_name, data))
            else:
                numeric_modalities[mod_name] = data

        # 2. Export CSV Telemetry
        self._save_csv(export_dir / "telemetry.csv", numeric_modalities, length, ep_id)

        # 3. Export Videos
        for mod_name, frames in visual_modalities:
            self._save_video(export_dir / f"{mod_name}.mp4", frames, mod_name, fps)

        log.info(f"✅ Export completed for episode {index}")

    def _save_csv(self, path: Path, data_map: Dict[str, np.ndarray], length: int, ep_id: str):
        log.info(f"📝 Writing telemetry to CSV...")
        
        # Flatten structure: Each field in each modality gets its own column
        rows = []
        for t in range(length):
            row = {"episode_id": ep_id, "step": t}
            for mod_name, arr in data_map.items():
                val = arr[t]
                if isinstance(val, (np.ndarray, list)):
                    if len(np.shape(val)) == 0: # scalar array
                         try: row[mod_name] = float(val)
                         except: row[mod_name] = str(val)
                    else:
                        for i, v in enumerate(val):
                            try: row[f"{mod_name}_{i}"] = float(v)
                            except: row[f"{mod_name}_{i}"] = str(v)
                else:
                    try:
                        row[mod_name] = float(val)
                    except (ValueError, TypeError):
                        row[mod_name] = str(val)
            rows.append(row)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            if not rows: return
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"   Saved: {path}")

    def _save_video(self, path: Path, frames: np.ndarray, mod_name: str, fps: int):
        log.info(f"🎞️ Generating video for {mod_name}...")
        
        if frames.dtype == np.uint8 and len(frames.shape) == 1:
            # This is likely still compressed bytes in a numpy array (happens if _get_full_modality_array isn't used right)
            # But ExpertTrajectoryDataset._get_full_modality_array handles jpeg decompression.
            # Let's double check the shape.
            pass

        # If data is still JPEG bytes (e.g. if we bypassed decompression)
        # But our ds helper should have returned [T, H, W, 3]
        if len(frames.shape) != 4:
            log.warning(f"   Skipping {mod_name}: Invalid shape {frames.shape}")
            return

        T, H, W, C = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))

        for t in range(T):
            frame = frames[t]
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        log.info(f"   Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="PhD-Grade Episode Exporter (CSV + MP4)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to LMDB file")
    parser.add_argument("--index", type=int, default=0, help="Episode index to export")
    parser.add_argument("--output_dir", type=str, default="output_data/exports", help="Root output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    args = parser.parse_args()

    try:
        exporter = EpisodeExporter(args.data_path, args.output_dir)
        exporter.export_episode(args.index, args.fps)
    except Exception as e:
        log.error(f"🚨 Export Failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
