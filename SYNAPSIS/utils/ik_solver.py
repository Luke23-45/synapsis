# utils/ik_solver.py
import numpy as np
from typing import List, Tuple, Optional
import ikpy.chain
from ikpy.link import URDFLink, OriginLink
import io
import logging 
from scipy.spatial.transform import Rotation as R
from mujoco import mj_jac  # For Jacobian
import mujoco
from numpy.linalg import inv, pinv, cond


logger = logging.getLogger(__name__) 


class IKSolver:
    """
    IKPy wrapper tuned for the Panda URDF that:
      - uses 'link0' as base element
      - detects ACTIVE links precisely (only revolute URDFLink entries)
      - exposes compute_action(target_pose_7d, current_joint_angles, max_delta)
        which returns normalized [-1,1] deltas for the arm (7 joints) + 1 gripper
    """

    def __init__(self, urdf_path: str, expect_7_dof: bool = True, kp=15.0, ki=0.1, kd=1.5,
                 lookahead_steps=2, ref_dist_pos=0.02, ref_dist_rot=0.1, max_boost=4.0):
        logger.info(f"⏳ [IKSolver] Loading kinematic chain from: {urdf_path}")
        try:
            # Load full chain starting from "link0"
            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=["link0"]
            )
            tcp_virtual_link = URDFLink(
                name="attachment_site_virtual_link",
                origin_translation=[0, 0, 0.1],  # 10cm Z-offset
                origin_orientation=[0, 0, 0],    # No rotation
                joint_type="fixed",
            )
            
            # Adaptive params (new)
            self.lookahead_steps = lookahead_steps
            self.ref_dist_pos = ref_dist_pos
            self.ref_dist_rot = ref_dist_rot
            self.max_boost = max_boost
            
            # Base gains (your originals)
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.damping = 2e-2
            self.reset_controller_state()
            self._integral_error = np.zeros(6)

            # 3. Create a new chain by appending the virtual link to the original's list of links.
            #    This is the correct way to modify the chain.
            self.chain = ikpy.chain.Chain(
                self.chain.links + [tcp_virtual_link]
            )

            # Trim chain until we hit the "hand_joint"
            self.chain.active_links_mask = [
                False, # link0 (base) is not active
                True,  # joint1
                True,  # joint2
                True,  # joint3
                True,  # joint4
                True,  # joint5
                True,  # joint6
                True,  # joint7
                False,  # hand_joint (fixed) is not active
                False
            ]
            
            self._active_idx = [i for i, flag in enumerate(self.chain.active_links_mask) if flag]
            self._active_joint_names = tuple(self.chain.links[i].name for i in self._active_idx)

            # read joint limits (in same order as self._active_idx)
            self._joint_limits: List[Tuple[float, float]] = []
            for idx in self._active_idx:
                lo, hi = -np.inf, np.inf
                link = self.chain.links[idx]
                try:
                    bounds = getattr(link, "bounds", None)
                    if bounds is not None and len(bounds) == 2:
                        lo, hi = float(bounds[0]), float(bounds[1])
                    else:
                        limit = getattr(link, "limit", None)
                        if limit is not None:
                            lo = float(getattr(limit, "lower", lo))
                            hi = float(getattr(limit, "upper", hi))
                except Exception:
                    # keep defaults if anything unexpected
                    lo, hi = -np.inf, np.inf
                self._joint_limits.append((lo, hi))

            # debug prints
            logger.info("Kinematic chain loaded and configured.")
            logger.debug(f"End effector (last link): {self.chain.links[-1].name}")
            logger.debug(f"Found {len(self._active_idx)} active joints: {self._active_joint_names}")

            if expect_7_dof and len(self._active_idx) != 7:
                logger.warning(f"Expected 7 active joints but found {len(self._active_idx)}.")

        except Exception as e:
            logger.critical("Could not parse URDF or build chain.", exc_info=True)
            raise



    def compute_action(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        max_delta: float = 0.1,  # This parameter is no longer used but kept for API consistency
        solution_position_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Computes a normalized POSITION action to move towards a target pose.

        This is the definitive, correct version. It solves for the final target joint angles
        and normalizes them to the action space [-1, 1]. The underlying MuJoCo
        simulation's position controller is responsible for executing the motion.

        Returns:
            A NumPy array of shape (7,) representing the normalized target joint positions.
            Returns the current joint positions (a "hold" command) on IK failure.
        """
        n_active = len(self._active_idx)

        # 1. Solve for the final target joint configuration.
        target_joint_angles = self._get_target_joint_angles(
            target_pose_7d,
            current_joint_angles,
            solution_position_tolerance,
        )

        # 2. Handle IK failure: command a "hold position" action.
        if target_joint_angles is None:
            logger.warning("IK solver failed. Commanding a hold action (current joint positions).")
            target_joint_angles = current_joint_angles

        # 3. Normalize the absolute target joint angles to the action space [-1, 1].
        action = np.zeros(n_active, dtype=np.float32)
        for i in range(n_active):
            lo, hi = self._joint_limits[i]
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                action[i] = np.clip(target_joint_angles[i], -1.0, 1.0)
                continue
            
            # Scale to [0, 1]
            scaled_pos = (target_joint_angles[i] - lo) / (hi - lo)
            # Scale to [-1, 1]
            action[i] = 2.0 * scaled_pos - 1.0

        # Clip to ensure it's strictly within the action space bounds.
        final_action = np.clip(action, -1.0, 1.0)

        if not np.all(np.isfinite(final_action)):
            logger.warning("IK produced a non-finite action. Returning zeros to prevent crash.")
            return np.zeros(n_active, dtype=np.float32)

        return final_action

    
    def joint_names(self) -> Tuple[str, ...]:
        """Return the active joint names in order."""
        return self._active_joint_names

    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp an N-dof vector (active joints) to URDF joint limits."""
        q = np.asarray(q, dtype=float).reshape(-1)
        n_active = len(self._active_idx)
        if q.shape != (n_active,):
            raise ValueError(f"Input must have shape ({n_active},)")
        out = np.empty_like(q)
        for i, val in enumerate(q):
            lo, hi = self._joint_limits[i]
            lo_use = -1e9 if not np.isfinite(lo) else lo
            hi_use = 1e9 if not np.isfinite(hi) else hi
            out[i] = np.clip(val, lo_use, hi_use)
        return out
    
    # Removed unused stubs compute_delta_action_ and compute_delta_action__1 for clarity


    def reset_controller_state(self):
        """Resets the internal state of the PID controller and fuzzy scheduler."""
        self._integral_error = np.zeros(6)
        self._prev_error = np.zeros(6)
        self._d_filter_state = np.zeros(6)
        
        # Feedforward state
        self._prev_target_pose = None
        self._first_step = True # Flag to prevent derivative kick
        
        # Fuzzy scheduler state
        self._fuzzy_prev_error = np.zeros(6)
        self._fuzzy_error_rate = np.zeros(6)
        self._fuzzy_error_rate_filtered = np.zeros(6)
        self._precision_mode_counter = 0

    def set_gains(self, kp: float, ki: float, kd: float,damping: float = 2e-2, lookahead_steps: Optional[int] = None,
                  ref_dist_pos: Optional[float] = None, ref_dist_rot: Optional[float] = None,
                  max_boost: Optional[float] = None):
        """
        Dynamically sets the PID gains for the controller.
        This is primarily used for tuning scripts. Optional adaptive params for enhancement.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.damping = damping
        if lookahead_steps is not None:
            self.lookahead_steps = lookahead_steps
        if ref_dist_pos is not None:
            self.ref_dist_pos = ref_dist_pos
        if ref_dist_rot is not None:
            self.ref_dist_rot = ref_dist_rot
        if max_boost is not None:
            self.max_boost = max_boost

    # ========================================================================================
    # [SOTA ENHANCEMENT] FUZZY GAIN SCHEDULING SYSTEM
    # Based on 2024 research: Mamdani-style inference with Gaussian membership functions
    # Provides phase-aware control: stability for large errors, precision for small errors
    # ========================================================================================

    def _gaussian_membership(self, x: float, center: float, sigma: float) -> float:
        """Gaussian membership function for fuzzy sets."""
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def _sigmoid_membership(self, x: float, center: float, steepness: float = 10.0) -> float:
        """Sigmoid membership function for smooth transitions."""
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    def fuzzy_gain_schedule_lef(
        self, 
        pos_error: np.ndarray, 
        rot_error_vec: np.ndarray, 
        effective_dt: float,
        phase_hint: Optional[str] = None  # Optional: "approach", "grasp", "place"
    ) -> Tuple[float, float]:
        """
        [PRODUCTION-GRADE] Fuzzy logic gain scheduling for Franka Panda manipulator.
        
        Implements Mamdani-style fuzzy inference with 4 rules optimized for BC/DGPO training:
        
        Rules:
        1. IF error is LARGE and velocity is HIGH -> Kp is LOW (Stability)
           - Prevents overshoot during fast approach
        2. IF error is LARGE and velocity is LOW  -> Kp is MEDIUM (Responsiveness)
           - Accelerates from stopped position toward target
        3. IF error is SMALL and velocity is HIGH -> Kp is LOW (Braking)
           - Prevents oscillation near target
        4. IF error is SMALL and velocity is LOW  -> Kp is VERY HIGH (Precision)
           - Maximum precision for fine placement
        
        Returns:
            Tuple[pos_boost, rot_boost]: Separate gain multipliers for position and rotation
        """
        # --- 1. COMPUTE NORMALIZED ERRORS ---
        pos_dist = np.linalg.norm(pos_error)
        rot_dist = np.linalg.norm(rot_error_vec)
        
        # Normalize to [0, ~3] range where 1.0 = reference distance
        e_pos_norm = pos_dist / self.ref_dist_pos  # ref_dist_pos = 0.01m
        e_rot_norm = rot_dist / self.ref_dist_rot  # ref_dist_rot = 0.1rad
        
        # --- 2. COMPUTE VELOCITY (ERROR RATE) WITH FILTERING ---
        curr_error = np.concatenate([pos_error, rot_error_vec])
        raw_error_rate = (curr_error - self._fuzzy_prev_error) / max(effective_dt, 1e-6)
        
        # Low-pass filter on error rate to reduce noise (alpha = 0.3)
        alpha_rate = 0.3
        self._fuzzy_error_rate_filtered = (
            alpha_rate * raw_error_rate + 
            (1 - alpha_rate) * self._fuzzy_error_rate_filtered
        )
        
        # Separate velocity norms
        v_pos = np.linalg.norm(self._fuzzy_error_rate_filtered[:3])
        v_rot = np.linalg.norm(self._fuzzy_error_rate_filtered[3:])
        
        # Normalize velocities (expected velocity = ref_dist / typical_dt)
        expected_vel_pos = self.ref_dist_pos / 0.01  # 1 m/s baseline
        expected_vel_rot = self.ref_dist_rot / 0.01  # 10 rad/s baseline
        v_pos_norm = v_pos / expected_vel_pos
        v_rot_norm = v_rot / expected_vel_rot
        
        # Update previous error for next iteration
        self._fuzzy_prev_error = curr_error.copy()
        
        # --- 3. MEMBERSHIP FUNCTIONS (GAUSSIAN) ---
        # Position Error Membership
        mu_pos_small = self._gaussian_membership(e_pos_norm, center=0.0, sigma=0.5)
        mu_pos_large = 1.0 - mu_pos_small
        
        # Rotation Error Membership  
        mu_rot_small = self._gaussian_membership(e_rot_norm, center=0.0, sigma=0.5)
        mu_rot_large = 1.0 - mu_rot_small
        
        # Velocity Membership
        mu_vel_pos_low = self._gaussian_membership(v_pos_norm, center=0.0, sigma=0.5)
        mu_vel_pos_high = 1.0 - mu_vel_pos_low
        mu_vel_rot_low = self._gaussian_membership(v_rot_norm, center=0.0, sigma=0.5)
        mu_vel_rot_high = 1.0 - mu_vel_rot_low
        
        # --- 4. FUZZY RULE INFERENCE (MAMDANI) ---
        # Define boost values for each rule
        BOOST_STABILITY = 1.5      # Rule 1: Large error + High velocity
        BOOST_RESPONSIVE = 4.0     # Rule 2: Large error + Low velocity
        BOOST_BRAKING = 1.0        # Rule 3: Small error + High velocity
        BOOST_PRECISION = self.max_boost  # Rule 4: Small error + Low velocity (max 10x)
        
        # Position gain calculation
        w1_pos = mu_pos_large * mu_vel_pos_high  # Stability
        w2_pos = mu_pos_large * mu_vel_pos_low   # Responsive
        w3_pos = mu_pos_small * mu_vel_pos_high  # Braking
        w4_pos = mu_pos_small * mu_vel_pos_low   # Precision
        
        total_w_pos = w1_pos + w2_pos + w3_pos + w4_pos + 1e-8
        pos_boost = (
            w1_pos * BOOST_STABILITY +
            w2_pos * BOOST_RESPONSIVE +
            w3_pos * BOOST_BRAKING +
            w4_pos * BOOST_PRECISION
        ) / total_w_pos
        
        # Rotation gain calculation (separate for finer control)
        w1_rot = mu_rot_large * mu_vel_rot_high
        w2_rot = mu_rot_large * mu_vel_rot_low
        w3_rot = mu_rot_small * mu_vel_rot_high
        w4_rot = mu_rot_small * mu_vel_rot_low
        
        total_w_rot = w1_rot + w2_rot + w3_rot + w4_rot + 1e-8
        rot_boost = (
            w1_rot * BOOST_STABILITY +
            w2_rot * BOOST_RESPONSIVE +
            w3_rot * BOOST_BRAKING +
            w4_rot * BOOST_PRECISION
        ) / total_w_rot
        
        # --- 5. PRECISION MODE HYSTERESIS ---
        # If we've been in precision mode for multiple consecutive steps,
        # lock the high gains to prevent oscillation from mode switching
        if pos_dist < self.ref_dist_pos * 0.5 and rot_dist < self.ref_dist_rot * 0.5:
            self._precision_mode_counter = min(self._precision_mode_counter + 1, 10)
        else:
            self._precision_mode_counter = max(self._precision_mode_counter - 2, 0)
        
        # Boost precision gains if we've been stable near target
        if self._precision_mode_counter >= 5:
            pos_boost = max(pos_boost, BOOST_PRECISION * 0.8)
            rot_boost = max(rot_boost, BOOST_PRECISION * 0.8)
        
        # --- 6. PHASE-AWARE ADJUSTMENTS (Optional) ---
        if phase_hint == "grasp":
            # During grasp, prioritize position precision over rotation
            pos_boost = min(pos_boost * 1.2, self.max_boost)
        elif phase_hint == "place":
            # During place, both position and rotation need high precision
            pos_boost = min(pos_boost * 1.1, self.max_boost)
            rot_boost = min(rot_boost * 1.1, self.max_boost)
        
        # --- 7. FINAL CLIP AND RETURN ---
        pos_boost = float(np.clip(pos_boost, 1.0, self.max_boost))
        rot_boost = float(np.clip(rot_boost, 1.0, self.max_boost))
        
        # Debug logging for tuning
        if pos_dist < 0.005:
            logger.debug(
                f"Fuzzy Gains: pos_boost={pos_boost:.2f}x (err={pos_dist*1000:.1f}mm) | "
                f"rot_boost={rot_boost:.2f}x (err={np.degrees(rot_dist):.1f}°) | "
                f"precision_mode={self._precision_mode_counter}"
            )
        
        return pos_boost, rot_boost

    def fuzzy_gain_schedule(
        self, 
        pos_error: np.ndarray, 
        rot_error_vec: np.ndarray, 
        effective_dt: float,
        phase_hint: Optional[str] = None
    ) -> Tuple[float, float]:
        
        # 1. Compute Distances
        pos_dist = np.linalg.norm(pos_error)
        rot_dist = np.linalg.norm(rot_error_vec)
        
        # 2. Compute Velocity (Error Rate)
        curr_error = np.concatenate([pos_error, rot_error_vec])
        # Simple finite diff is sufficient
        raw_error_rate = (curr_error - self._fuzzy_prev_error) / max(effective_dt, 1e-6)
        
        # Update state
        self._fuzzy_prev_error = curr_error.copy()
        
        v_pos = np.linalg.norm(raw_error_rate[:3])

        # --- RE-ENGINEERED RULES ---
        
        # State A: FAR AWAY (Approach Phase)
        # We need SPEED. Base Kp (15.0) is too slow. Boost it to ~45.0.
        if pos_dist > 0.10: # > 10cm
            return 3.0, 3.0 # Boost 3x -> Kp = 45.0

        # State B: CLOSE PROXIMITY (The Danger Zone)
        # Error is < 2cm. 
        # OLD LOGIC: Boost to 4.0x (Kp=60). Result: Vibration.
        # NEW LOGIC: Boost lightly (Kp=30) but rely on Kd for braking.
        if pos_dist < 0.02: 
            # If moving fast near target, keep gains moderate to prevent overshoot
            if v_pos > 0.1: 
                return 1.5, 1.5 # Kp = 22.5
            # If stopped near target, stiffen slightly to hold
            else:
                return 2.0, 2.0 # Kp = 30.0

        # State C: TRANSITION (2cm - 10cm)
        # Linear interpolation or standard boost
        return 2.5, 2.5

    def compute_target_joint_positions(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        solution_position_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        [ROBUST ANALYTICAL IK]
        Returns the raw target joint angles (radians) needed to reach the pose.
        Uses warm-starting from current_joint_angles.
        Returns start angles if IK fails.
        """
        target_joint_angles = self._get_target_joint_angles(
            target_pose_7d,
            current_joint_angles,
            solution_position_tolerance,
        )
        if target_joint_angles is None:
            # Fallback to holding position
            return current_joint_angles.copy()
        return target_joint_angles


    def compute_dynamic_delta_action(
        self,
        target_ee_pose_chunk: np.ndarray, 
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        joint_qpos_indices: np.ndarray,
        effective_dt: float,
        max_dq: float,
    ) -> np.ndarray:
        """
        [FIXED DELTA CONTROLLER]
        Includes Internal Feedforward, Derivative Kick protection, and Tuned Fuzzy Gains.
        """
        # --- 1. PARSE TARGET & CALCULATE FEEDFORWARD VELOCITY ---
        if target_ee_pose_chunk.ndim == 1:
            target_ee_pose = target_ee_pose_chunk
        else:
            idx = min(self.lookahead_steps, len(target_ee_pose_chunk) - 1)
            target_ee_pose = target_ee_pose_chunk[idx]

        # INTERNAL FEEDFORWARD CALCULATION
        ff_vel = np.zeros(6)
        if self._prev_target_pose is not None and effective_dt > 1e-6:
            # Linear Velocity FF
            ff_pos = (target_ee_pose[:3] - self._prev_target_pose[:3]) / effective_dt
            
            # Angular Velocity FF
            q_curr = R.from_quat(self._prev_target_pose[3:])
            q_next = R.from_quat(target_ee_pose[3:])
            dq_rot = q_next * q_curr.inv()
            ff_rot = dq_rot.as_rotvec() / effective_dt
            
            ff_vel = np.concatenate([ff_pos, ff_rot])
            ff_vel = np.clip(ff_vel, -1.0, 1.0) # Clamp noise

        self._prev_target_pose = target_ee_pose.copy()

        # --- 2. GET CURRENT ROBOT STATE ---
        current_ee_pos = data.site_xpos[ee_site_id]
        current_ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        
        if np.abs(np.linalg.det(current_ee_mat)) < 1e-6:
            current_ee_quat = np.array([0, 0, 0, 1], dtype=np.float64)
        else:
            current_ee_quat = R.from_matrix(current_ee_mat).as_quat()

        # Jacobian
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        ee_body_id = model.site_bodyid[ee_site_id]
        mujoco.mj_jac(model, data, jac_pos, jac_rot, current_ee_pos, ee_body_id)
        J_full = np.vstack([jac_pos, jac_rot])
        J = J_full[:, joint_qpos_indices]

        # --- 3. COMPUTE ERRORS ---
        pos_error = target_ee_pose[:3] - current_ee_pos
        target_quat = R.from_quat(target_ee_pose[3:])
        current_quat = R.from_quat(current_ee_quat)
        orn_error_vec = (target_quat * current_quat.inv()).as_rotvec()
        error_6d = np.concatenate([pos_error, orn_error_vec])

        # --- 4. GAINS ---
        pos_boost, rot_boost = self.fuzzy_gain_schedule(pos_error, orn_error_vec, effective_dt)
        
        base_kp = 50.0 
        kp_vec = np.array([base_kp * pos_boost] * 3 + [base_kp * rot_boost] * 3)
        
        base_kd = 4.0
        kd_vec = np.array([base_kd * np.sqrt(pos_boost)] * 3 + [base_kd * np.sqrt(rot_boost)] * 3)
        
        base_ki = 5.0
        ki_vec = np.array([base_ki] * 6)

        # --- 5. PID CALCULATION ---
        
        # P-Term: Feedback + Feedforward
        p_term = (kp_vec * error_6d) + ff_vel

        # D-Term: Derivative Kick Protection & Smoothing
        if self._first_step:
            self._prev_error = error_6d.copy()
            self._d_filter_state = np.zeros(6)
            self._first_step = False
            
        error_deriv = (error_6d - self._prev_error) / effective_dt
        
        # [TUNING] Increased time constant to 0.03 (30ms) to reduce Step 30+ jitter
        tau_d = 0.03  
        alpha = effective_dt / (tau_d + effective_dt)
        filtered_deriv = (1 - alpha) * self._d_filter_state + alpha * error_deriv
        
        d_term = kd_vec * filtered_deriv

        # I-Term
        i_term = ki_vec * self._integral_error

        # Total Cartesian Velocity Demand
        ee_vel_target = p_term + d_term + i_term

        # --- 6. IK SOLVE ---
        damping = 0.05 
        try:
            lhs = J.T @ J + damping * np.eye(J.shape[1])
            rhs = J.T @ ee_vel_target
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.zeros(len(joint_qpos_indices))

        # --- 7. OUTPUT & STATE UPDATE ---
        action_raw = dq / max_dq
        
        # Anti-Windup
        is_saturated = np.any(np.abs(action_raw) > 0.95)
        if not is_saturated:
            self._integral_error += error_6d * effective_dt
            np.clip(self._integral_error, -0.05, 0.05, out=self._integral_error)
            
        if np.linalg.norm(pos_error) > 0.05:
            self._integral_error *= 0.9

        self._prev_error = error_6d.copy()
        self._d_filter_state = filtered_deriv.copy()

        return np.clip(action_raw, -1.0, 1.0)

    def solve_ik_static(
        self,
        target_pose: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        q0: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Public API for Static Inverse Kinematics.
        Returns raw joint angles (qpos) for a given 7D target pose.
        Useful for setting the robot state directly (e.g., for goal image generation).
        
        Args:
            target_pose: (7,) [x, y, z, qx, qy, qz, qw]
            model: MuJoCo model
            data: MuJoCo data
            ee_site_id: End-effector site ID
            q0: Initial guess for joint angles (seed)
            
        Returns:
            (7,) np.ndarray of joint angles, or None if IK fails.
        """
        # _get_target_joint_angles expects current_joint_angles as the seed
        # It handles the multi-attempt logic and returns raw qpos (float32)
        return self._get_target_joint_angles(
            target_pose_7d=target_pose,
            current_joint_angles=q0,
            solution_position_tolerance=0.01
        )


    @staticmethod
    def _rot_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        """Quaternion [w, x, y, z] -> 3x3 rotation matrix (right-handed)."""
        w, x, y, z = map(float, q_wxyz)
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n == 0.0:
            return np.eye(3, dtype=float)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )  


    def _get_target_joint_angles(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        solution_position_tolerance: float,
        max_iter: int = 100, # Reduced max_iter for speed
        regularization_strength: float = 1e-4
    ) -> Optional[np.ndarray]:
        """
        Solves the inverse kinematics problem with a robust, multi-attempt fallback strategy.
        """
        current_joint_angles = self.clamp_to_limits(current_joint_angles)
        target_pos = target_pose_7d[:3]
        target_orient_matrix = R.from_quat(target_pose_7d[3:]).as_matrix()

        initial_guess = [0.0] * len(self.chain.links)
        for i, joint_val in enumerate(current_joint_angles):
            initial_guess[self._active_idx[i]] = joint_val

        # --- START OF PATCH: Multi-Attempt IK Solving ---
        
        # Define our attempts, from most to least constrained
        orientation_modes = ["all", "Z", None]
        
        for i, mode in enumerate(orientation_modes):
            solved_joints_full = None # Reset solution for this attempt
            try:
                solved_joints_full = self.chain.inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_orient_matrix if mode is not None else None,
                    orientation_mode=mode,
                    initial_position=initial_guess,
                    max_iter=max_iter,
                    regularization_parameter=regularization_strength
                )
            except Exception:
                # This attempt failed entirely, continue to the next one
                continue

            # If a solution was found, verify its quality
            if solved_joints_full is not None and np.all(np.isfinite(solved_joints_full)):
                fk_frame = self.chain.forward_kinematics(solved_joints_full)
                result_pos = fk_frame[:3, 3]
                position_error = np.linalg.norm(result_pos - target_pos)

                # If the solution is accurate enough, we are done!
                if position_error <= solution_position_tolerance:
                    active_solved_joints = [solved_joints_full[i] for i in self._active_idx]
                    return np.array(active_solved_joints, dtype=np.float32)
        
        # If all attempts failed to produce an accurate solution, return None
        logger.warning(f"IK solver failed on all attempts to reach {np.round(target_pos, 2)}.")
        return None
        # --- END OF PATCH ---



    def compute_osc_action(
        self,
        target_ee_pose_chunk: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        joint_qpos_indices: np.ndarray,
        effective_dt: float,
        max_dq: float = 3.0, # [FINAL SPEED UNLOCK]
        posture_q_desired: np.ndarray = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    ) -> np.ndarray:
        """
        [FINAL GOLDEN DELTA - PRODUCTION RELEASE]
        The empirically verified best controller.
        
        Architecture:
        - Kinematic Damped Least Squares (Implicitly stable).
        - High-Gain PID (Precision).
        - Unrestricted Feedforward (Speed).
        - NO Nullspace Projection (Eliminates drift).
        """
        n_arm = len(joint_qpos_indices)

        # --- 1. TARGETING (Lookahead=1) ---
        if target_ee_pose_chunk.ndim == 1:
            target_ee_pose = target_ee_pose_chunk
        else:
            idx = min(1, len(target_ee_pose_chunk) - 1)
            target_ee_pose = target_ee_pose_chunk[idx]

        # --- 2. FEEDFORWARD (Unrestricted) ---
        ff_vel = np.zeros(6)
        if self._prev_target_pose is not None and effective_dt > 1e-6:
            ff_pos = (target_ee_pose[:3] - self._prev_target_pose[:3]) / effective_dt
            q_curr = R.from_quat(self._prev_target_pose[3:])
            q_next = R.from_quat(target_ee_pose[3:])
            dq_rot = q_next * q_curr.inv()
            ff_rot = dq_rot.as_rotvec() / effective_dt
            ff_vel = np.concatenate([ff_pos, ff_rot])
            
            # [SPEED] Allow full physical velocity (5.0 m/s)
            ff_vel = np.clip(ff_vel, -5.0, 5.0)

        self._prev_target_pose = target_ee_pose.copy()

        # --- 3. STATE ---
        ee_pos = data.site_xpos[ee_site_id]
        ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        if np.abs(np.linalg.det(ee_mat)) < 1e-6:
            ee_quat_xyzw = np.array([0, 0, 0, 1.0])
        else:
            ee_quat_xyzw = R.from_matrix(ee_mat).as_quat()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        J = np.vstack([jacp, jacr])[:, joint_qpos_indices]

        # --- 4. ERROR ---
        pos_error = target_ee_pose[:3] - ee_pos
        target_quat = R.from_quat(target_ee_pose[3:])
        current_quat = R.from_quat(ee_quat_xyzw)
        rot_error_vec = (target_quat * current_quat.inv()).as_rotvec()
        error_6d = np.concatenate([pos_error, rot_error_vec])

        # --- 5. GAINS (The "Golden" Set) ---
        # These provided the best stability. Do not over-tune.
        pos_boost, rot_boost = self.fuzzy_gain_schedule(pos_error, rot_error_vec, effective_dt)
        
        base_kp = 50.0 
        kp_vec = np.array([base_kp * pos_boost] * 3 + [base_kp * rot_boost] * 3)
        
        base_kd = 4.0
        kd_vec = np.array([base_kd * np.sqrt(pos_boost)] * 3 + [base_kd * np.sqrt(rot_boost)] * 3)
        
        base_ki = 5.0
        ki_vec = np.array([base_ki] * 6)

        # --- 6. PID COMPUTATION ---
        p_term = (kp_vec * error_6d) + ff_vel

        if self._first_step:
            self._prev_error = error_6d.copy()
            self._d_filter_state = np.zeros(6)
            self._first_step = False
            
        error_deriv = (error_6d - self._prev_error) / effective_dt
        tau_d = 0.02 # Fast filter
        alpha = effective_dt / (tau_d + effective_dt)
        filtered_deriv = (1 - alpha) * self._d_filter_state + alpha * error_deriv
        d_term = kd_vec * filtered_deriv

        i_term = ki_vec * self._integral_error

        # Velocity Command
        v_task = p_term + d_term + i_term
        
        # [SPEED] Relaxed Clamp
        v_norm = np.linalg.norm(v_task)
        if v_norm > 5.0:
            v_task = v_task * (5.0 / v_norm)

        # --- 7. SOLVER (Implicit Regularization) ---
        # Standard DLS handles singularities and redundancy naturally without drift.
        damping = 0.05 
        try:
            # (J^T J + lambda I) dq = J^T v
            lhs = J.T @ J + damping * np.eye(J.shape[1])
            rhs = J.T @ v_task
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.zeros(n_arm)

        # --- 8. OUTPUT ---
        action_raw = dq / max_dq
        
        is_saturated = np.any(np.abs(action_raw) > 0.95)
        if not is_saturated:
            self._integral_error += error_6d * effective_dt
            np.clip(self._integral_error, -0.1, 0.1, out=self._integral_error)
            
        if np.linalg.norm(pos_error) > 0.05:
            self._integral_error *= 0.9

        self._prev_error = error_6d.copy()
        self._d_filter_state = filtered_deriv.copy()

        return np.clip(action_raw, -1.0, 1.0)
    def normalize_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Normalize physical joint angles (radians) to the action space [-1, 1].
        
        Args:
            q: (N,) array of physical joint positions.
            
        Returns:
            (N,) array of normalized values in range [-1, 1].
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        n_active = len(self._active_idx)
        
        # Robust handling if q includes extra joints (e.g. gripper)
        if q.shape[0] > n_active:
            q = q[:n_active]
        elif q.shape[0] < n_active:
            raise ValueError(f"Input joints shape {q.shape} < active joints {n_active}")

        out = np.zeros(n_active, dtype=np.float32)
        
        for i in range(n_active):
            val = q[i]
            lo, hi = self._joint_limits[i]
            
            # Handle unbounded or fixed joints
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                out[i] = np.clip(val, -1.0, 1.0)
                continue
            
            # Clamp to physical limits first
            val = np.clip(val, lo, hi)
            
            # Map [lo, hi] -> [0, 1]
            scaled = (val - lo) / (hi - lo)
            # Map [0, 1] -> [-1, 1]
            out[i] = 2.0 * scaled - 1.0
            
        return out  
  
    def compute_delta_action(
        self,
        target_ee_pose: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        joint_qpos_indices: np.ndarray,
        effective_dt: float,
        max_dq: float,
    ) -> np.ndarray:
        """
        [DEFINITIVE, TUNED FOR Kp=40, PRODUCTION-GRADE PID CONTROLLER V4]
        This version uses a balanced set of gains for a responsive yet stable
        system, based on the manual tuning methodology.
        - Kp=40.0 provides a fast response.
        - Kd=4.0 provides critical damping to prevent overshoot.
        - Ki=1.0 provides slow, stable correction for steady-state error.
        """
        # --- 1. GET CURRENT STATE ---
        current_ee_pos = data.site_xpos[ee_site_id]
        current_ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        
        # SAFETY: Handle null/invalid rotation matrices
        mat_det = np.linalg.det(current_ee_mat)
        if np.abs(mat_det) < 1e-6:
            # Null or degenerate matrix - use identity quaternion
            logger.warning("IK: Null rotation matrix detected, using identity orientation")
            current_ee_quat = np.array([0, 0, 0, 1], dtype=np.float64)  # xyzw identity
        else:
            current_ee_quat = R.from_matrix(current_ee_mat).as_quat()
        
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        ee_body_id = model.site_bodyid[ee_site_id]
        mujoco.mj_jac(model, data, jac_pos, jac_rot, current_ee_pos, ee_body_id)
        
        J_full = np.vstack([jac_pos, jac_rot])
        J = J_full[:, joint_qpos_indices]

        # --- 2. IMPLEMENT STABLE & TUNED PID CONTROL LAW ---
        # FINAL TUNED GAINS for Kp=40
        # Kp = 321.4
        # Kd = 28.4
        # Ki =  5.37
        # integral_clamp = 0.4
        # damping = 1e-2

        # print(f"current: kp {self.kp} - kd-> {self.kd} - ki->{self.ki}")

        # Kp = self.kp
        # Kd = self.kd
        # Ki = self.ki

        Kp = 139.0  
        Kd = 3.0    
        Ki = 1.0    # [TUNED] Increased from 0.1 to eliminate steady-state error

        integral_clamp = 0.5 # [TUNED] Increased from 0.1 to allow sufficient gravity compensation
        damping = 2e-2

        # Filter for the derivative term to prevent noise amplification
        tau_d = 3.0 * effective_dt # Derivative filter time constant
        alpha = effective_dt / (tau_d + effective_dt)

        # Calculate 6D error vector
        pos_error = target_ee_pose[:3] - current_ee_pos
        orn_error_vec = (R.from_quat(target_ee_pose[3:]) * R.from_quat(current_ee_quat).inv()).as_rotvec()
        error_6d = np.concatenate([pos_error, orn_error_vec])

        # Proportional Term
        p_term = Kp * error_6d

        # Derivative Term (on error, and filtered)
        prev_error = getattr(self, "_prev_error", np.zeros_like(error_6d))
        error_deriv = (error_6d - prev_error) / effective_dt
        
        prev_filtered_deriv = getattr(self, "_d_filter_state", np.zeros_like(error_deriv))
        filtered_deriv = (1 - alpha) * prev_filtered_deriv + alpha * error_deriv
        d_term = Kd * filtered_deriv

        # Integral Term (with conditional anti-windup)
        integrator = getattr(self, "_integral_error", np.zeros_like(error_6d))
        i_term = Ki * integrator

        # Target velocity is the sum of PID components
        ee_vel_target = p_term + i_term + d_term

        # --- 3. SOLVE FOR JOINT VELOCITIES (dIK) ---
        try:
            lhs = J.T @ J + damping * np.eye(J.shape[1])
            rhs = J.T @ ee_vel_target
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.zeros(len(joint_qpos_indices))

        # --- 4. ANTI-WINDUP & NORMALIZE ACTION ---
        action_if_applied = dq / max_dq
        is_saturated = np.any(np.abs(action_if_applied) > 1.0)

        # Conditional Integration: Only integrate if the controller is not saturated.
        if not is_saturated:
            integrator += error_6d * effective_dt
            np.clip(integrator, -integral_clamp, integral_clamp, out=integrator)
        
        # Store states for next step
        self._prev_error = error_6d.copy()
        self._d_filter_state = filtered_deriv.copy()
        self._integral_error = integrator.copy()

        action = np.clip(action_if_applied, -1.0, 1.0)
        return action