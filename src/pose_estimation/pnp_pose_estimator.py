"""
pnp_pose_estimator.py

Pose estimation for a preceding vehicle using the Perspective-n-Point (PnP) algorithm.

This is the main pose estimation module used by the pipeline. It uses cv2.solvePnP
(SQPNP method) with up to 6 2D-3D correspondences to estimate the 6-DoF pose.

For comparison/diagnostic purposes, a separate VP-based method is available in
vp_pose_estimator.py (implements the original Task 2 vanishing-point formulation).

Algorithm overview
------------------
1. Feature collection  – gather up to 6 2D/3D correspondences (tail-light
                         outer edges, top/bottom corners, outer midpoint).
2. PnP solve           – ``cv2.solvePnP`` (SQPNP) returns rvec and tvec that
                         map the vehicle object frame to the camera frame.
3. Reprojection guard  – reject poses whose mean reprojection error exceeds a
                         configurable pixel threshold (tracker drift filter).
4. Motion VP correction – the X-axis of R is refined using the vanishing point
                          of inter-frame optical flow, correcting for systematic
                          bias introduced by a laterally offset / tilted camera.
5. Temporal smoothing  – EMA on tvec, SLERP on R to reduce frame-to-frame jitter.
6. Derived quantities  – distance, yaw, TTI (Time To Impact).

Why tvec comes directly from solvePnP
--------------------------------------
The earlier geometric formula  Z = W * f / pixel_dist  was replaced because:
  1. It assumes the lights are perpendicular to the camera line of sight, which
     is false when the camera is mounted laterally or at a tilt.
  2. It conflates depth Z (along the optical axis) with Euclidean distance.
  3. With relative yaw between camera and vehicle, apparent light separation
     varies → Z is systematically overestimated → bounding-box too small.

solvePnP accounts for the full camera geometry and returns the correct tvec
from which  distance = ‖tvec‖  regardless of camera placement.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Union, List
from collections import deque


class PnPPoseEstimator:
    """
    Estimates the 6-DoF pose of a preceding vehicle from tail-light features.

    Inputs expected per frame (``features_2d`` dict):
        'outer'       : (2, 2) array – left and right outer light centers
        'top'         : (2, 2) array – left and right top light corners
        'bottom'      : (2, 2) array – left and right bottom light corners
        'plate_bottom': (2, 2) array – bottom-left and bottom-right plate corners

    Outputs (returned dict from ``estimate_pose_multifeature``):
        'rvec', 'tvec', 'R', 'distance', 'motion_type', 'tti', 'vy', 'vx', ...
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        vehicle_model: dict,
        distortion_coeffs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            camera_matrix:    3×3 intrinsic matrix K.
            vehicle_model:    Configuration dict (from vehicle_model.yaml).
            distortion_coeffs: OpenCV distortion coefficients; zeros if None.
        """
        self.K     = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        self.dist_coeffs = (distortion_coeffs if distortion_coeffs is not None
                            else np.zeros(5, dtype=np.float32))

        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        # --- 3-D model of PnP correspondence points (vehicle frame, metres) ---
        pnp_cfg = vehicle_model.get('vehicle', {}).get('pnp_points_3d', {})
        self.map_3d = {
            'origin':   np.array([0.0, 0.0, 0.0],                            dtype=np.float32),
            'l_outer':  np.array(pnp_cfg.get('light_l_outer'),               dtype=np.float32),
            'r_outer':  np.array(pnp_cfg.get('light_r_outer'),               dtype=np.float32),
            'l_top':    np.array(pnp_cfg.get('light_l_top'),                 dtype=np.float32),
            'r_top':    np.array(pnp_cfg.get('light_r_top'),                 dtype=np.float32),
            'l_bottom': np.array(pnp_cfg.get('light_l_bottom'),              dtype=np.float32),
            'r_bottom': np.array(pnp_cfg.get('light_r_bottom'),              dtype=np.float32),
        }

        # Pre-compute derived geometric constants
        self.center_outer_3d       = (self.map_3d['l_outer'] + self.map_3d['r_outer']) / 2.0
        self.lights_to_origin_offset = self.map_3d['origin'] - self.center_outer_3d
        self.lights_distance_real  = float(np.linalg.norm(
            self.map_3d['r_outer'] - self.map_3d['l_outer']
        ))
        self.lights_height = self.center_outer_3d[2]

        # Optional uniform scale applied to the PnP tvec output (tune in [0.80, 1.0]
        # if systematic distance over/underestimation is observed after calibration).
        self.tvec_scale = 1.0

        # --- Vehicle dimensions (used by callers for bounding-box projection) ---
        vehicle_data = vehicle_model.get('vehicle', {})
        dimensions = vehicle_data.get('dimensions', {})
        self.vehicle_length = dimensions.get('length', 3.70)
        self.vehicle_width  = dimensions.get('width',  1.74)
        self.vehicle_height = dimensions.get('height', 1.525)

        # --- Temporal smoothing buffers ---
        self.tvec_history = deque(maxlen=5)
        self.rvec_history = deque(maxlen=5)
        self.yaw_history  = deque(maxlen=20)

        self.yaw_smooth   = None
        self.alpha_yaw    = 0.25   # EMA factor for yaw (0 = frozen, 1 = no smoothing)
        self.prev_yaw     = 0.0

        # EMA factors: higher → faster response but more jitter
        self.alpha_translation = 0.65
        self.alpha_rotation    = 0.35

        self.prev_tvec_smooth = None
        self.prev_R_smooth    = None

        # --- Reprojection-error-based pose rejection ---
        self.max_reproj_error  = 150.0  # pixels; poses above this are discarded
        self.last_reproj_error = 0.0    # exposed for external debugging

        # --- TTI (Time To Impact) state ---
        self.tti_min       = 0.5   # seconds; smaller values rejected as implausible
        self.tti_max       = 30.0  # seconds; larger values also rejected
        self.prev_distance = None

        # --- Motion vanishing point (Vx) state ---
        # Vx is the image-plane point toward which all inter-frame flow vectors
        # converge; it encodes the vehicle's forward direction in the camera frame.
        self.prev_features_2d = None
        self.vx_smooth        = None
        self.alpha_vx         = 0.35   # EMA factor for Vx
        self.vx_weight        = 0.0    # blend weight: 0 = pure PnP, 1 = pure Vx
        self.vx_min_points    = 3      # minimum flow vectors needed to estimate Vx
        self.vx_min_movement  = 1.0    # pixel threshold: discard near-static points
        self.vy_history = deque(maxlen=5)

        # --- Rotation classification state ---
        self.yaw_translation_threshold  = np.radians(8)
        self.rotation_angle_threshold   = np.radians(12)
        self.persistence_frames         = 3
        self.reference_x_axis           = None
        self.rotation_counter           = 0
        self.translation_counter        = 0

        self.prev_center_3d = None

        self.debug_mode = True

        print(f"[VanishingPointSolver] Initialized")
        print(f"  Outer-to-outer real distance : {self.lights_distance_real:.3f} m")
        print(f"  Smoothing: EMA(alpha_t={self.alpha_translation}) + "
              f"SLERP(alpha_r={self.alpha_rotation})")

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    def _ensure_numpy_array(
        self,
        data: Union[Tuple, List, np.ndarray],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Convert a tuple, list, or ndarray to a float32 array of the given shape."""
        if isinstance(data, (tuple, list)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
        if arr.shape != shape:
            arr = arr.reshape(shape)
        return arr

    # ------------------------------------------------------------------
    # Approximate distance estimate (fallback only)
    # ------------------------------------------------------------------

    def estimate_distance_from_outer_points(self, outer_points: np.ndarray) -> float:
        """
        Rough distance estimate from the apparent width of the tail lights.

        This formula  Z = W * f / pixel_dist  is kept as a lightweight fallback
        or sanity check only. It is NOT used in the main pose pipeline because:
          - It assumes the lights face the camera perpendicularly (breaks with yaw).
          - Z is depth along the optical axis, not Euclidean distance.
          - Both errors grow when the camera is offset laterally from the vehicle.

        The accurate distance is  ‖tvec_pnp‖  from ``reconstruct_pose_robust``.

        Args:
            outer_points: (2, 2) array [[lx, ly], [rx, ry]].

        Returns:
            Estimated depth in metres, clamped to [2, 50], mean reprojection error.
        """
        L, R = outer_points[0], outer_points[1]
        pixel_dist = np.linalg.norm(R - L)
        if pixel_dist < 1.0:
            return 10.0  # default when lights overlap (very close vehicle)
        f_avg = (self.fx + self.fy) / 2.0
        Z = (self.lights_distance_real * f_avg) / pixel_dist
        return float(np.clip(Z, 2.0, 50.0))

    # ------------------------------------------------------------------
    # Yaw estimation helpers
    # ------------------------------------------------------------------

    def estimate_yaw_from_lights_geometry(
        self, outer_points: np.ndarray, distance: float
    ) -> float:
        """
        Estimate vehicle yaw from the apparent positions of the two outer lights.

        The 2-D pixel positions are back-projected to 3-D rays scaled by
        ``distance``, and the resulting light-to-light vector is projected onto
        the horizontal plane to recover the yaw angle.

        Args:
            outer_points: (2, 2) array [[lx, ly], [rx, ry]] (image pixels).
            distance:     Estimated vehicle depth in metres.

        Returns:
            Yaw angle in radians (0 = vehicle facing straight ahead).
        """
        L_2d, R_2d = outer_points[0], outer_points[1]

        # Ensure left-right ordering
        if L_2d[0] > R_2d[0]:
            L_2d, R_2d = R_2d, L_2d

        L_ray = self.K_inv @ np.append(L_2d, 1.0)
        R_ray = self.K_inv @ np.append(R_2d, 1.0)
        L_3d = (L_ray / np.linalg.norm(L_ray)) * distance
        R_3d = (R_ray / np.linalg.norm(R_ray)) * distance

        # Project the inter-light vector onto the horizontal (XZ) plane
        y_dir_cam  = R_3d - L_3d
        y_horiz    = np.array([y_dir_cam[0], 0.0, y_dir_cam[2]])
        norm       = np.linalg.norm(y_horiz)
        if norm < 1e-6:
            return self.prev_yaw

        y_horiz     = y_horiz / norm
        y_veh_horiz = -y_horiz                                    # vehicle Y points left→right
        yaw = np.arctan2(y_veh_horiz[0], y_veh_horiz[2]) + np.pi / 2

        # Wrap to [-π, π]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        return float(yaw)

    def estimate_yaw_from_plate_bottom(
        self, plate_bottom: np.ndarray, distance: float
    ) -> float:
        """
        Estimate vehicle yaw from the bottom edge of the license plate.

        Functionally identical to ``estimate_yaw_from_lights_geometry`` but
        uses the plate's bottom-left / bottom-right points, which can provide
        a more stable yaw estimate at large distances (plate edge is wider
        relative to image resolution than individual light blobs).

        Args:
            plate_bottom: (2, 2) array [[BL_x, BL_y], [BR_x, BR_y]].
            distance:     Estimated vehicle depth in metres.

        Returns:
            Yaw angle in radians.
        """
        BL, BR = plate_bottom[0], plate_bottom[1]

        BL_ray = self.K_inv @ np.append(BL, 1.0)
        BR_ray = self.K_inv @ np.append(BR, 1.0)
        BL_3d = (BL_ray / np.linalg.norm(BL_ray)) * distance
        BR_3d = (BR_ray / np.linalg.norm(BR_ray)) * distance

        y_dir_cam = BR_3d - BL_3d
        y_horiz   = np.array([y_dir_cam[0], 0.0, y_dir_cam[2]])
        norm      = np.linalg.norm(y_horiz)
        if norm < 1e-6:
            return self.prev_yaw

        y_horiz     = y_horiz / norm
        y_veh_horiz = -y_horiz
        yaw = np.arctan2(y_veh_horiz[0], y_veh_horiz[2]) + np.pi / 2

        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        return float(yaw)

    # ------------------------------------------------------------------
    # Rotation math utilities
    # ------------------------------------------------------------------

    def slerp_rotation(
        self, R1: np.ndarray, R2: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Spherical Linear Interpolation (SLERP) between two rotation matrices.

        SLERP interpolates along the geodesic on SO(3), preserving the constant
        angular velocity property. This is preferable to naive linear blending
        (which does not stay on SO(3)) for smoothing rotation estimates.

        Args:
            R1: Starting rotation matrix (3×3).
            R2: Ending rotation matrix (3×3).
            t:  Interpolation factor in [0, 1]; 0 → R1, 1 → R2.

        Returns:
            Interpolated rotation matrix (3×3).
        """
        q1 = self._rotation_to_quaternion(R1)
        q2 = self._rotation_to_quaternion(R2)

        # Ensure shortest-path interpolation
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2  = -q2
            dot = -dot

        dot   = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-6:
            # Quaternions nearly identical → linear blend is accurate enough
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (
                (np.sin((1 - t) * theta) / np.sin(theta)) * q1
                + (np.sin(t * theta)     / np.sin(theta)) * q2
            )

        q_interp /= np.linalg.norm(q_interp)
        return self._quaternion_to_rotation(q_interp)

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

        Uses Shepperd's method with branch selection based on the dominant
        diagonal element to avoid numerical instability near degenerate cases.
        """
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
        ])

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def apply_temporal_smoothing(
        self,
        tvec_raw: np.ndarray,
        R_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply per-frame temporal smoothing to the pose estimate.

        Translation uses EMA (Exponential Moving Average) because tvec lives
        in a Euclidean space where linear blending is meaningful.

        Rotation uses SLERP because SO(3) is not a Euclidean space; naively
        averaging rotation matrices does not in general produce a valid rotation.

        Args:
            tvec_raw: Raw translation vector (3,) from solvePnP.
            R_raw:    Raw rotation matrix (3×3) from solvePnP.

        Returns:
            (tvec_smooth, R_smooth) – smoothed pose.
        """
        if self.prev_tvec_smooth is None:
            tvec_smooth = tvec_raw.copy()
        else:
            tvec_smooth = (self.alpha_translation * tvec_raw
                           + (1 - self.alpha_translation) * self.prev_tvec_smooth)

        if self.prev_R_smooth is None:
            R_smooth = R_raw.copy()
        else:
            R_smooth = self.slerp_rotation(self.prev_R_smooth, R_raw, self.alpha_rotation)

        self.prev_tvec_smooth = tvec_smooth.copy()
        self.prev_R_smooth    = R_smooth.copy()

        return tvec_smooth, R_smooth

    # ------------------------------------------------------------------
    # Vanishing point computation
    # ------------------------------------------------------------------

    def compute_motion_vp(
        self,
        features_curr: dict,
        features_prev: dict,
    ) -> Optional[np.ndarray]:
        """
        Compute the motion vanishing point (Vx) from inter-frame optical flow.

        Vx is the image point toward which all flow vectors p_prev → p_curr
        converge. Because the vehicle is a rigid body moving on a flat road,
        all tracked points share the same Vx, so SVD on the homogeneous line
        system gives a robust, outlier-tolerant estimate.

        Only 'outer' and 'top' feature groups are used; 'bottom' is excluded
        because bottom-left tends to flicker and degrades the estimate.

        Args:
            features_curr: Feature dict for the current frame.
            features_prev: Feature dict for the previous frame.

        Returns:
            (vx, vy) image-plane coordinates of the motion VP, or None if
            insufficient moving points are available.
        """
        lines = []

        for key in ('outer', 'top'):
            if key not in features_curr or key not in features_prev:
                continue
            curr = np.array(features_curr[key], dtype=np.float64)
            prev = np.array(features_prev[key], dtype=np.float64)

            for i in range(min(len(curr), len(prev))):
                movement = np.linalg.norm(curr[i] - prev[i])
                if movement < self.vx_min_movement:
                    continue  # stationary point – does not contribute to Vx

                # Homogeneous line through the two points
                p0 = np.array([prev[i][0], prev[i][1], 1.0])
                p1 = np.array([curr[i][0], curr[i][1], 1.0])
                line = np.cross(p0, p1)
                norm = np.linalg.norm(line)
                if norm > 1e-8:
                    lines.append(line / norm)

        # Add the outer-centre midpoint as an extra, more stable flow point
        if 'outer' in features_curr and 'outer' in features_prev:
            c_curr = np.mean(np.array(features_curr['outer'], dtype=np.float64), axis=0)
            c_prev = np.mean(np.array(features_prev['outer'], dtype=np.float64), axis=0)
            if np.linalg.norm(c_curr - c_prev) >= self.vx_min_movement:
                p0   = np.array([c_prev[0], c_prev[1], 1.0])
                p1   = np.array([c_curr[0], c_curr[1], 1.0])
                line = np.cross(p0, p1)
                norm = np.linalg.norm(line)
                if norm > 1e-8:
                    lines.append(line / norm)

        if len(lines) < self.vx_min_points:
            return None

        # The motion VP is the null-space vector of the line system (SVD last row)
        _, _, Vt = np.linalg.svd(np.array(lines))
        vp_h = Vt[-1]

        if abs(vp_h[2]) < 1e-8:
            return None

        vp = vp_h[:2] / vp_h[2]

        # Reject degenerate solutions (VP at infinity → near-zero motion)
        if np.linalg.norm(vp) > 1e5:
            return None

        return vp

    def compute_lateral_vp(self, features_2d: dict) -> Optional[np.ndarray]:
        """
        Compute the lateral vanishing point (Vy) from same-frame L-R point pairs.

        Vy is the image point where lines connecting left-right homologous
        features (outer L-R, top L-R, plate bottom BL-BR) converge. It
        encodes the direction of the vehicle's Y-axis (width axis) in the
        image plane, and is independent of inter-frame motion.

        Args:
            features_2d: Feature dict for a single frame.

        Returns:
            (vx, vy) image-plane coordinates of the lateral VP, or None if
            fewer than two valid L-R pairs are available.
        """
        lines = []
        pairs = []

        if 'outer' in features_2d:
            o = features_2d['outer']
            pairs.append((o[0], o[1]))

        if 'top' in features_2d:
            t = features_2d['top']
            pairs.append((t[0], t[1]))

        if 'plate_bottom' in features_2d:
            pb = features_2d['plate_bottom']
            if len(pb) == 2:
                pairs.append((pb[0], pb[1]))

        for pL, pR in pairs:
            pLh  = np.array([float(pL[0]), float(pL[1]), 1.0])
            pRh  = np.array([float(pR[0]), float(pR[1]), 1.0])
            line = np.cross(pLh, pRh)
            norm = np.linalg.norm(line)
            if norm > 1e-8:
                lines.append(line / norm)

        if len(lines) < 2:
            return None

        _, _, Vt = np.linalg.svd(np.array(lines))
        vp_h = Vt[-1]

        if abs(vp_h[2]) < 1e-8:
            return None

        vp = vp_h[:2] / vp_h[2]

        if np.linalg.norm(vp) > 1e6:
            return None

        return vp

    # ------------------------------------------------------------------
    # Rotation correction
    # ------------------------------------------------------------------

    def correct_rotation_with_vp(
        self, R_pnp: np.ndarray, vx: np.ndarray
    ) -> np.ndarray:
        """
        Refine the PnP rotation matrix using the motion vanishing point.

        PnP can estimate the X-axis (vehicle forward direction) with a
        systematic bias when the camera is laterally offset or tilted, because
        the 3-D model assumes a known camera position that may not match
        reality precisely. The motion VP gives the true forward direction
        directly from image-plane flow, without any 3-D model assumption.

        Reconstruction strategy:
            X  ← back-projected VP direction (from flow)
            Y  ← kept from PnP (lateral axis, well-constrained by light geometry)
            Z  ← X × Y (enforces right-hand orthogonality)
            Y' ← Z × X (re-orthogonalise to remove numerical drift)

        The final result is blended between R_pnp and the VP-derived R using
        ``self.vx_weight`` (0 = pure PnP, 1 = pure VP).

        Args:
            R_pnp: Rotation matrix from solvePnP (3×3).
            vx:    Motion vanishing point in image coordinates (2,).

        Returns:
            Corrected (blended) rotation matrix (3×3).
        """
        # Back-project the VP into a 3-D camera-frame direction
        x_from_vp = self.K_inv @ np.array([vx[0], vx[1], 1.0])
        x_from_vp /= np.linalg.norm(x_from_vp)

        y_pnp = R_pnp[:, 1] / np.linalg.norm(R_pnp[:, 1])

        z_corrected = np.cross(x_from_vp, y_pnp)
        norm_z = np.linalg.norm(z_corrected)
        if norm_z < 1e-6:
            return R_pnp  # degenerate configuration → fall back to PnP

        z_corrected /= norm_z
        y_corrected  = np.cross(z_corrected, x_from_vp)
        y_corrected /= np.linalg.norm(y_corrected)

        R_from_vp = np.column_stack([x_from_vp, y_corrected, z_corrected])

        return self.slerp_rotation(R_pnp, R_from_vp, self.vx_weight)

    # ------------------------------------------------------------------
    # Core pose reconstruction
    # ------------------------------------------------------------------

    def reconstruct_pose_robust(
        self, features_2d: dict, frame_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate vehicle pose via direct PnP with up to 6 correspondences.

        Correspondence set
        ------------------
        Up to 6 2D–3D pairs are assembled:
            l_outer, r_outer  – always used (minimum viable set)
            l_top,   r_top    – used when available
            r_bottom           – used when available
            center outer       – synthetic midpoint (improves convergence)

        Distance note
        -------------
        The returned tvec comes directly from solvePnP.
        Do NOT recompute distance from pixel width; see module docstring.

        Args:
            features_2d: Feature dict with keys 'outer', 'top', 'bottom'.
            frame_idx:   Current frame index (used for periodic debug prints).

        Returns:
            (rvec_smooth, tvec_smooth, R_smooth) on success, or (None, None, None)
            if PnP failed or the reprojection error exceeds the rejection threshold.
        """
        outer_points = features_2d.get('outer')
        if outer_points is None or len(outer_points) < 2:
            return None, None, None, None

        # --- Build 2D–3D correspondence lists ---

        def to_point_list(arr):
            """Ensure each point is a plain Python float tuple (required by solvePnP)."""
            if isinstance(arr, np.ndarray):
                return [tuple(map(float, pt)) for pt in arr]
            return arr

        outer_pts  = to_point_list(features_2d.get('outer',  [None, None]))
        top_pts    = to_point_list(features_2d.get('top',    [None, None]))
        bottom_pts = to_point_list(features_2d.get('bottom', [None, None]))

        mapping = [
            (outer_pts[0] if len(outer_pts) > 0 else None, 'l_outer'),
            (outer_pts[1] if len(outer_pts) > 1 else None, 'r_outer'),
            (top_pts[0]   if len(top_pts)   > 0 else None, 'l_top'),
            (top_pts[1]   if len(top_pts)   > 1 else None, 'r_top'),
        ]

        # r_bottom is included only when the tracker provides it reliably
        if len(bottom_pts) > 1 and bottom_pts[1] is not None:
            mapping.append((bottom_pts[1], 'r_bottom'))

        list_2d, list_3d = [], []
        for pt_2d, key_3d in mapping:
            if pt_2d is not None:
                list_2d.append(pt_2d)
                list_3d.append(self.map_3d[key_3d])

        # 6th point: synthetic outer midpoint (adds a constraint on depth)
        if (len(outer_pts) >= 2
                and outer_pts[0] is not None
                and outer_pts[1] is not None):
            center_2d = tuple(map(float, np.mean([outer_pts[0], outer_pts[1]], axis=0)))
            center_3d = (self.map_3d['l_outer'] + self.map_3d['r_outer']) / 2.0
            list_2d.append(center_2d)
            list_3d.append(center_3d)

        if len(list_2d) < 4:
            return None, None, None

        img_pts = np.array(list_2d, dtype=np.float32)
        obj_pts = np.array(list_3d, dtype=np.float32)

        success, rvec_pnp, tvec_pnp = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            return None, None, None, None

        # --- Reprojection error check ---
        # High reprojection error signals tracker drift (stale 2-D points).
        # Rejecting such poses prevents jumps in downstream distance / TTI estimates.
        projected, _ = cv2.projectPoints(obj_pts, rvec_pnp, tvec_pnp,
                                          self.K, self.dist_coeffs)
        mean_reproj_error = float(np.mean(
            np.linalg.norm(img_pts - projected.reshape(-1, 2), axis=1)
        ))
        self.last_reproj_error = mean_reproj_error

        if mean_reproj_error > self.max_reproj_error:
            if self.debug_mode:
                print(f"  [Frame {frame_idx}] Pose rejected: "
                      f"reproj_error={mean_reproj_error:.1f} px "
                      f"> threshold {self.max_reproj_error} px")
            return None, None, None, None

        R_pnp, _ = cv2.Rodrigues(rvec_pnp)

        # --- Motion VP correction for the X-axis ---
        # PnP with a laterally mounted camera can produce a systematic X-axis
        # error. The motion VP, computed from consecutive-frame flow, gives the
        # true forward direction without assuming a known camera extrinsic pose.
        if self.prev_features_2d is not None:
            vx_raw = self.compute_motion_vp(features_2d, self.prev_features_2d)

            if vx_raw is not None:
                if self.vx_smooth is None:
                    self.vx_smooth = vx_raw
                else:
                    delta = vx_raw - self.vx_smooth
                    # Reject single-frame outliers (VP jump > 200 px)
                    if np.linalg.norm(delta) < 200.0:
                        self.vx_smooth = self.vx_smooth + self.alpha_vx * delta

                R_pnp = self.correct_rotation_with_vp(R_pnp, self.vx_smooth)

                if self.debug_mode and frame_idx % 10 == 0:
                    print(f"  [Frame {frame_idx}] Vx_smooth="
                          f"({self.vx_smooth[0]:.0f}, {self.vx_smooth[1]:.0f}), "
                          f"vx_weight={self.vx_weight}")

        # Store features for next frame's flow computation
        self.prev_features_2d = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in features_2d.items()
        }

        # --- Apply optional scale correction and temporal smoothing ---
        tvec_final = tvec_pnp.reshape(3, 1).astype(np.float32) * self.tvec_scale
        tvec_smooth, R_smooth = self.apply_temporal_smoothing(
            tvec_final.flatten(), R_pnp
        )
        rvec_smooth, _ = cv2.Rodrigues(R_smooth)
        tvec_smooth    = tvec_smooth.reshape(3, 1).astype(np.float32)

        if self.debug_mode and frame_idx % 10 == 0:
            pitch = np.arctan2(-R_pnp[2, 1], R_pnp[2, 2])
            dist  = float(np.linalg.norm(tvec_pnp))
            dist_h = float(np.sqrt(float(tvec_pnp[0])**2 + float(tvec_pnp[2])**2))
            print(f"  [Frame {frame_idx}] pitch={np.degrees(pitch):.1f}°, "
                  f"dist={dist:.2f} m (horiz={dist_h:.2f} m), n_pts={len(list_2d)}")

        return rvec_smooth, tvec_smooth, R_smooth, mean_reproj_error

    # ------------------------------------------------------------------
    # Yaw extraction and motion classification
    # ------------------------------------------------------------------

    def extract_yaw_from_rotation(self, R: np.ndarray) -> float:
        """
        Extract the yaw angle (rotation around the vertical axis) from R.

        The X-column of R gives the vehicle's forward direction in the camera
        frame. Projecting it onto the horizontal plane and taking the
        arctangent yields the yaw.

        Args:
            R: 3×3 rotation matrix.

        Returns:
            Yaw in radians.
        """
        x_axis    = R[:, 0].copy()
        x_axis[1] = 0.0  # project onto horizontal plane
        norm      = np.linalg.norm(x_axis)
        if norm < 1e-6:
            return 0.0
        x_axis /= norm
        return float(np.arctan2(x_axis[0], x_axis[2]))

    def classify_motion_type(self, R: np.ndarray) -> str:
        """
        Classify vehicle motion as 'TRANSLATION' or 'STEERING' based on
        the range of smoothed yaw values over a rolling 20-frame window.

        A yaw range > 5° across recent frames indicates the vehicle is turning.

        Args:
            R: Current smoothed rotation matrix.

        Returns:
            'STEERING' or 'TRANSLATION'.
        """
        raw_yaw = self.extract_yaw_from_rotation(R)

        # EMA smoothing with wrap-around handling for ±π boundary
        if self.yaw_smooth is None:
            self.yaw_smooth = raw_yaw
        else:
            diff = raw_yaw - self.yaw_smooth
            if diff >  np.pi: diff -= 2 * np.pi
            if diff < -np.pi: diff += 2 * np.pi
            self.yaw_smooth += self.alpha_yaw * diff

        self.yaw_history.append(self.yaw_smooth)

        if len(self.yaw_history) < 3:
            return "TRANSLATION"

        yaw_range = max(self.yaw_history) - min(self.yaw_history)
        return "STEERING" if yaw_range > np.radians(5) else "TRANSLATION"

    # ------------------------------------------------------------------
    # TTI (Time To Impact)
    # ------------------------------------------------------------------

    def calculate_tti(self, distance: float, dt: float) -> Optional[float]:
        """
        Estimate Time To Impact using finite differences on successive distances.

        TTI = -distance / closing_velocity  (positive when vehicle is approaching)

        Implausible velocity jumps (> 5 m/s between frames) are rejected.

        Args:
            distance: Current vehicle distance in metres.
            dt:       Time elapsed since the previous frame in seconds.

        Returns:
            TTI in seconds, or None if insufficient data or implausible velocity.
        """
        if self.prev_distance is None:
            self.prev_distance = distance
            return None

        V = (distance - self.prev_distance) / dt
        self.prev_distance = distance

        # Sanity checks: reject velocity outliers and non-approaching scenarios
        if abs(V) > 5.0:
            return None
        if V >= -0.01:
            return None  # vehicle is stationary or moving away

        return float(-distance / V)

    def validate_pose_with_tti(self, tti: Optional[float]) -> Tuple[bool, str]:
        """
        Validate a pose estimate by checking whether the implied TTI is physical.

        Args:
            tti: Time To Impact in seconds (may be None).

        Returns:
            (is_valid, reason_string)
        """
        if tti is None:
            return True, "TTI not available"
        if tti < 0:
            if abs(tti) > self.tti_max:
                return False, f"Moving away too slowly (TTI={tti:.1f}s)"
            return True, f"Moving away (TTI={tti:.1f}s)"
        if tti < self.tti_min:
            return False, f"TTI too small (TTI={tti:.2f}s)"
        if tti > self.tti_max:
            return False, f"TTI too large (TTI={tti:.1f}s)"
        return True, f"TTI valid ({tti:.1f}s)"

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_tti_history(self):
        """Clear the distance history used for TTI computation."""
        self.prev_distance = None

    def reset_temporal_smoothing(self):
        """Clear all temporal smoothing buffers (call when tracking is lost)."""
        self.vy_history.clear()
        self.yaw_history.clear()
        self.tvec_history.clear()
        self.rvec_history.clear()
        self.prev_tvec_smooth = None
        self.prev_R_smooth    = None
        self.prev_features_2d = None
        self.vx_smooth        = None

    def reset_vp_persistence(self):
        """Full reset: clears all state (use when vehicle re-enters the scene)."""
        self.prev_center_3d   = None
        self.prev_yaw         = 0.0
        self.reference_x_axis = None
        self.rotation_counter    = 0
        self.translation_counter = 0
        self.reset_temporal_smoothing()
        print("[VanishingPointSolver] State reset.")

    # ------------------------------------------------------------------
    # Public estimation entry points
    # ------------------------------------------------------------------

    def estimate_pose_multifeature(
        self,
        features_t2: Dict[str, np.ndarray],
        plate_bottom_t2: Optional[np.ndarray],
        frame_idx: int = 0,
    ) -> Optional[Dict]:
        """
        Main estimation entry point.

        Assembles all available features, runs the PnP pipeline, and returns
        a rich result dictionary with pose, distance, motion type, TTI, and
        both vanishing points.

        Args:
            features_t2:     Feature dict for the current frame (keys: 'outer',
                             'top', 'bottom').
            plate_bottom_t2: (2, 2) plate bottom corners [BL, BR], or None.
            frame_idx:       Current frame index.

        Returns:
            Result dict or None if pose estimation failed.
        """
        if features_t2.get('outer') is None:
            return None

        # Work on a copy so callers can safely modify features_t2 afterwards
        features = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in features_t2.items()
        }
        if plate_bottom_t2 is not None:
            features['plate_bottom'] = plate_bottom_t2

        rvec, tvec, R, mean_reproj_error  = self.reconstruct_pose_robust(features, frame_idx)
        if rvec is None:
            return None

        vy_lateral = self.compute_lateral_vp(features)
        vx_motion  = self.vx_smooth  # already EMA-smoothed in reconstruct_pose_robust

        distance = float(np.linalg.norm(tvec))
        motion_type = self.classify_motion_type(R)
        tti = self.calculate_tti(distance, dt=1.0 / 30.0)
        tti_valid, tti_msg = self.validate_pose_with_tti(tti)

        yaw_deg = (np.degrees(self.yaw_smooth)
                   if self.yaw_smooth is not None
                   else np.degrees(self.extract_yaw_from_rotation(R)))

        return {
            'rvec':       rvec,
            'tvec':       tvec,
            'R':          R,
            'distance':   distance,
            'frame':      frame_idx,
            'is_valid':   True,
            'motion_type': motion_type,
            'reproj_error': mean_reproj_error,
            'tti':        tti,
            'tti_valid':  tti_valid,
            'vy':         vy_lateral,
            'vx':         vx_motion,
            'debug': {
                'method':         'pnp6_direct_tvec',
                'yaw_degrees':    yaw_deg,
                'pitch_degrees':  np.degrees(np.arctan2(-R[2, 1], R[2, 2])),
                'has_plate':      plate_bottom_t2 is not None,
                'smoothing':      (f"EMA(t={self.alpha_translation})"
                                   f"+SLERP(r={self.alpha_rotation})"),
            },
        }

    def estimate_pose(
        self,
        lights_frame: Union[Tuple, np.ndarray],
        plate_bottom: Optional[np.ndarray],
        frame_idx: int = 0,
    ) -> Optional[Dict]:
        """
        Backward-compatible single-feature entry point.

        Wraps ``estimate_pose_multifeature`` using only the outer light pair.
        Retained so that callers written for the older API continue to work
        without modification.

        Args:
            lights_frame: (2, 2) outer light positions [[lx, ly], [rx, ry]].
            plate_bottom: (2, 2) plate bottom corners, or None.
            frame_idx:    Current frame index.

        Returns:
            Same as ``estimate_pose_multifeature``.
        """
        features = {'outer': self._ensure_numpy_array(lights_frame, (2, 2))}
        return self.estimate_pose_multifeature(features, plate_bottom, frame_idx)