import argparse
import os
import cv2
import csv
import json
import logging
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime

# Ensure SYNAPSIS is importable
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Import your project modules
from SYNAPSIS.datasets.expert_dataset import ExpertDataset
from SYNAPSIS.envs.scripted_expert import ExpertConfig
from SYNAPSIS.envs.panda_env import DomainRandomizationConfig

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DataGen")

class EpisodeGenerator(ExpertDataset):
    """
    A specialized subclass of ExpertDataset that yields FULL episodes 
    for visualization/logging rather than individual samples for training.
    
    This reuses 100% of the SOTA 'Pass 1 (Physics) -> Pass 2 (Render)' logic
    to ensure the video matches the exact physics used in the dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_episodes(self) -> Iterator[Dict[str, Any]]:
        """
        Modified iterator that yields the complete 'episode_dict' 
        immediately after the rendering pass succeeds.
        """
        # Initialize worker state (RNG, Env, Expert, IK)
        self._init_worker_state()
        
        # We access the internal generation logic via the iterator
        # But we intercept the 'self.episodes' list which is populated inside __iter__
        
        consecutive_failures = 0
        MAX_CONSEC = 25
        
        while True:
            # Check limits
            if self.max_episodes_per_epoch and self._episode_id_counter >= self.max_episodes_per_epoch:
                return

            # Clear buffers
            self._episode_buffer.clear()
            
            try:
                # We need to clear self.episodes to ensure we get the fresh one
                self.episodes = [] 
                
                # Create an iterator for the parent class
                parent_iter = super().__iter__()
                
                # Pull *one* sample. This triggers the generation of a FULL episode 
                # (Pass 1 & Pass 2) inside the parent, populates self._episode_buffer, 
                # and appends to self.episodes.
                try:
                    _ = next(parent_iter)
                except StopIteration:
                    return

                # If we got here, an episode was successfully generated and stored.
                if self.episodes:
                    latest_episode = self.episodes[-1]
                    yield latest_episode
                    consecutive_failures = 0
                else:
                    # Should technically not happen if next() succeeded
                    consecutive_failures += 1

            except RuntimeError as e:
                logger.warning(f"Expert Generation Failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSEC:
                    logger.error("Too many consecutive failures. Aborting.")
                    return
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return

def save_video(episode: Dict[str, Any], output_path: Path, fps: int = 30):
    """
    Compiles a side-by-side video of Primary and Wrist cameras.
    """
    obs_list = episode["obs_list"]
    if not obs_list:
        return

    # Dimensions
    h_prim, w_prim, _ = obs_list[0]["image_primary"].shape
    h_wrist, w_wrist, _ = obs_list[0]["image_wrist"].shape
    
    # We resize wrist to match primary height for a clean side-by-side
    scale = h_prim / h_wrist
    new_w_wrist = int(w_wrist * scale)
    
    full_w = w_prim + new_w_wrist
    full_h = h_prim
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (full_w, full_h))
    
    for step in obs_list:
        # Convert RGB to BGR for OpenCV
        img_p = cv2.cvtColor(step["image_primary"], cv2.COLOR_RGB2BGR)
        img_w = cv2.cvtColor(step["image_wrist"], cv2.COLOR_RGB2BGR)
        
        # Resize wrist
        img_w_resized = cv2.resize(img_w, (new_w_wrist, full_h), interpolation=cv2.INTER_AREA)
        
        # Concatenate
        frame = np.hstack((img_p, img_w_resized))
        out.write(frame)
        
    out.release()

def flatten_data_for_csv(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flattens the nested episode dictionary into a list of row-dicts for CSV.
    """
    rows = []
    length = len(episode["actions"])
    
    for t in range(length):
        obs = episode["obs_list"][t]
        action = episode["actions"][t]
        ik_fail = episode["ik_fail_flags"][t]
        
        row = {
            "episode_id": episode["episode_id"],
            "seed": episode["seed"],
            "timestep": t,
            "ik_valid": not ik_fail,
            # Action (8 dims)
            "action_0": action[0], "action_1": action[1], "action_2": action[2],
            "action_3": action[3], "action_4": action[4], "action_5": action[5],
            "action_6": action[6], "action_gripper": action[7],
            # [FIX] Joint Angles (7 dims) - Required for Temporal Sampler analysis
            "joint_0": obs.get("arm_joints", [0]*7)[0],
            "joint_1": obs.get("arm_joints", [0]*7)[1],
            "joint_2": obs.get("arm_joints", [0]*7)[2],
            "joint_3": obs.get("arm_joints", [0]*7)[3],
            "joint_4": obs.get("arm_joints", [0]*7)[4],
            "joint_5": obs.get("arm_joints", [0]*7)[5],
            "joint_6": obs.get("arm_joints", [0]*7)[6],
            # EE Pose World (7 dims)
            "ee_x": obs["ee_pose_world"][0], "ee_y": obs["ee_pose_world"][1], "ee_z": obs["ee_pose_world"][2],
            "ee_qx": obs["ee_pose_world"][3], "ee_qy": obs["ee_pose_world"][4], 
            "ee_qz": obs["ee_pose_world"][5], "ee_qw": obs["ee_pose_world"][6],
            # SOTA Delta EE Pose (if available)
            "delta_x": obs.get("delta_ee_pose", [0]*7)[0],
            "delta_y": obs.get("delta_ee_pose", [0]*7)[1],
            "delta_z": obs.get("delta_ee_pose", [0]*7)[2],
            # Object Pose
            "obj_x": obs["object_pos_world"][0], "obj_y": obs["object_pos_world"][1], "obj_z": obs["object_pos_world"][2],
            # Goal Pose
            "goal_x": obs["goal_pos_world"][0], "goal_y": obs["goal_pos_world"][1], "goal_z": obs["goal_pos_world"][2],
            # Expert State
            "expert_phase": obs.get("gt_phase", [-1])[0],
            "gripper_intent": obs.get("gt_gripper", [-1.0])[0],
            "state_str": obs.get("expert_state", "UNKNOWN")
        }
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Generate SOTA Expert Trajectories (Video + CSV)")
    
    # Configuration
    parser.add_argument("--urdf_path", type=str, default="SYNAPSIS/envs/urdf/panda.urdf", help="Path to robot URDF")
    parser.add_argument("--xml_path", type=str, default="SYNAPSIS/envs/panda_pick_place.xml", help="Path to MuJoCo XML")
    parser.add_argument("--output_dir", type=str, default="output_data/debug", help="Directory to save outputs")
    
    # Generation Params
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to generate")
    parser.add_argument("--seed", type=int, default=43, help="Base seed for generation")
    parser.add_argument("--control_mode", type=str, default="delta", choices=["absolute", "delta"], 
                        help="Physics control mode")
    parser.add_argument("--recording_mode", type=str, default="cartesian_delta", 
                        choices=["cartesian_delta", "joint_absolute", "joint_delta"],
                        help="What action representation to record in the CSV/Dataset")
    
    # Expert Tweaks
    parser.add_argument("--no_filter", action="store_true", 
                        help="Disable velocity-based frame filtering for smoother videos (increases CSV size)")
    
    args = parser.parse_args()

    # 1. Setup Output Directory
    # out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    csv_dir = out_dir / "csvs"
    video_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    # 2. Configure Expert
    # To get a smooth video, we disable the 'sparse sampling' feature of the dataset
    # by setting probabilities to 1.0 if --no_filter is used.


    # scripted_cfg = ExpertConfig(
    #     max_grasp_retries=2,
    #     hover_height=0.15,
    #     cruise_velocity=0.3
    # )

    scripted_cfg = ExpertConfig()

    # 3. Instantiate Generator (Subclass)
    logger.info(f"Initializing Generator... (Control: {args.control_mode}, Record: {args.recording_mode})")
    
    # FIX 1: Removed erroneous 'max_dq' calculation here (variable 'env' does not exist in this scope)
    
    generator = EpisodeGenerator(
        urdf_path=args.urdf_path,
        env_xml_path=args.xml_path,
        base_seed=args.seed,
        max_episodes_per_epoch=args.num_episodes,
        control_mode=args.control_mode,
        recording_mode=args.recording_mode,
        scripted_cfg=scripted_cfg,

        
        warmup=True
    )

    # 4. Main Generation Loop
    count = 0
    logger.info("Starting generation loop...")
    
    try:
        # We use our custom generator method
        for episode in generator.generate_episodes():
            ep_id = episode["episode_id"]
            seed_used = episode["seed"]
            
            logger.info(f"Generated Episode {count+1}/{args.num_episodes} (ID: {ep_id}, Seed: {seed_used})")
            
            # --- Save Video ---
            vid_path = video_dir / f"{ep_id}_seed{seed_used}.mp4"
            save_video(episode, vid_path)
            
            # --- Save CSV ---
            csv_path = csv_dir / f"{ep_id}_seed{seed_used}.csv"
            rows = flatten_data_for_csv(episode)
            
            if rows:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            
            count += 1
            if count >= args.num_episodes:
                break
                
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user.")
    finally:
        logger.info(f"Done. Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()