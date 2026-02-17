"""
vehicle_localization_system.py

Main entry point for the tail-light-based vehicle tracking and
pose-estimation system.

On startup the script presents a two-option menu:
    1 - Recalibrate camera   – runs chessboard calibration from the images
                               stored in  data/calibration/images/  and writes
                               the result back to the calibration file
                               referenced in  config/camera_config.yaml.
    2 - Process video        – runs the full tracking pipeline on a user-
                               selected input video.
    0 - Exit

Processing pipeline overview
-----------------------------
1. Multi-feature tail-light detection  – outer, top, and bottom keypoints per
   light blob, plus the license plate bottom edge.
2. Hybrid tracking  – CSRT tracker per feature group, validated each frame by
   Lucas-Kanade optical flow, and periodically refined by template matching.
3. Re-detection  – triggered after consecutive tracking failures.
4. Outlier filtering  – median-motion consensus suppresses single-point jumps.
5. 6-DoF pose estimation  – PnP solver (SQPNP, up to 6 correspondences) with
   EMA + SLERP temporal smoothing.
6. Pose freeze  – the 3-D bounding box, origin, and axes are locked at a
   configurable frame index to avoid drift at the end of the sequence.
7. 3-D bounding-box projection and TTI (Time To Impact) estimation.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup – add src/ to the import search path
# ---------------------------------------------------------------------------

current_dir  = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from calibration.camera_calibration import CameraCalibrator
from calibration.load_calibration import (
    load_camera_calibration_simple as load_camera_calibration,
    CameraParameters,
)
from pose_estimation.vp_pose_estimator import VPPoseEstimator
from detection.light_detector import LightDetector
from detection.candidate_selector import CandidateSelector
from detection.advanced_detector import AdvancedDetector
from tracking.tracker import LightTracker
from tracking.redetection import RedetectionManager
from pose_estimation.pnp_pose_estimator import PnPPoseEstimator
from pose_estimation.bbox_3d_projector import BBox3DProjector
from utils.config_loader import load_config
from visualization.video_writer import VideoWriter
from visualization.draw_utils import (
    draw_tracking_info,
    draw_motion_type_overlay,
    draw_bbox_3d,
    draw_3d_axes,
    draw_vanishing_points_complete,
    # Note: Vx and ground-plane visualisation are included for completeness.
    # They are NOT used for pose or distance estimation because they are
    # unreliable in low-light / noisy nighttime scenes.
)
from visualization.draw_utils import DrawUtils


# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

VIDEO_DIR  = project_root / "data" / "videos" / "input"
OUTPUT_DIR = project_root / "data" / "videos" / "output"
CONFIG_DIR = project_root / "config"
CALIB_DIR  = project_root / "data" / "calibration"


# ===========================================================================
# Configuration loading
# ===========================================================================

def load_all_configs() -> dict:
    """
    Load all YAML configuration files required by the pipeline.

    Returns:
        Dict with keys 'camera_config', 'vehicle_model', 'detection_params'.

    Raises:
        FileNotFoundError: If any required config file is missing.
    """
    config = {}

    required = {
        'camera_config':    CONFIG_DIR / 'camera_config.yaml',
        'vehicle_model':    CONFIG_DIR / 'vehicle_model.yaml',
        'detection_params': CONFIG_DIR / 'detection_params.yaml',
    }

    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required configuration file not found: {path}")
        config[key] = load_config(str(path))

    return config


# ===========================================================================
# Camera recalibration
# ===========================================================================

def recalibrate_camera(config: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Run chessboard calibration from the images in  data/calibration/images/
    and overwrite the calibration file referenced in camera_config.yaml.

    The user is prompted to confirm (or override) the chessboard pattern
    dimensions and square size before calibration begins.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        (camera_matrix, dist_coeffs) on success, or None on failure.
    """
    image_dir = CALIB_DIR / "images"
    if not image_dir.exists():
        print(f"Calibration image directory not found: {image_dir}")
        return None

    image_files = sorted(
        list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
    )
    if not image_files:
        print(f"No calibration images found in {image_dir}")
        return None

    print(f"\nCalibration images available ({len(image_files)}):")
    for f in image_files:
        print(f"  {f.name}")

    print("\nChessboard pattern parameters (press Enter to accept defaults from config):")
    try:
        # Read defaults from camera_config.yaml so the script
        # stays in sync with the rest of the configuration.
        pat_cfg      = config['camera_config'].get('calibration', {}).get('pattern', {})
        default_size = pat_cfg.get('size', [7, 3])
        default_sq   = pat_cfg.get('square_size', 0.025)

        inp  = input(f"  Inner corners per row   [{default_size[0]}]: ").strip()
        cols = int(inp) if inp else default_size[0]

        inp  = input(f"  Inner corners per column [{default_size[1]}]: ").strip()
        rows = int(inp) if inp else default_size[1]

        inp         = input(f"  Square size in metres [{default_sq}]: ").strip()
        square_size = float(inp) if inp else default_sq
        
    except (ValueError, KeyboardInterrupt):
        print("Calibration cancelled.")
        return None

    # Resolve output path from config
    calib_file  = config['camera_config']['camera']['calibration_file']
    output_path = (Path(calib_file) if Path(calib_file).is_absolute()
                   else project_root / calib_file)

    print(f"\nRunning calibration with {len(image_files)} images...")
    try:
        calibrator = CameraCalibrator(
            pattern_size=(cols, rows),
            square_size=square_size,
        )
        for img_path in image_files:
            calibrator.add_image(str(img_path), visualize=False)

        camera_matrix, dist_coeffs, mean_error = calibrator.calibrate()
        calibrator.save_calibration(str(output_path), camera_matrix, dist_coeffs)

        print(f"Mean reprojection error : {mean_error:.4f} px")
        print(f"Calibration saved to    : {output_path}")
        return camera_matrix, dist_coeffs

    except Exception as e:
        print(f"Calibration failed: {e}")
        return None


# ===========================================================================
# Video selection helper
# ===========================================================================

def choose_video() -> Tuple[Optional[str], Optional[str]]:
    """
    List available input videos and prompt the user to select one.

    Returns:
        (video_path, video_name) or (None, None) on invalid selection.
    """
    if not VIDEO_DIR.exists():
        print(f"Video directory not found: {VIDEO_DIR}")
        return None, None

    videos = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ])
    if not videos:
        print(f"No video files found in {VIDEO_DIR}")
        return None, None

    print("\nAvailable videos:")
    for i, v in enumerate(videos, 1):
        print(f"  {i} - {v}")

    try:
        idx = int(input("\nSelect video number: ")) - 1
        if 0 <= idx < len(videos):
            video_name = videos[idx]
            return str(VIDEO_DIR / video_name), video_name
        print("Invalid selection.")
        return None, None
    except (ValueError, KeyboardInterrupt):
        return None, None


# ===========================================================================
# Tracking pipeline
# ===========================================================================

class VehicleTrackingPipeline:
    """
    Per-video state machine for multi-feature tail-light tracking and pose
    estimation.

    Architecture
    ------------
    Detection layer    – AdvancedDetector extracts three sub-points per light
                         (outer, top, bottom) and the plate bottom edge.
    Tracking layer     – One CSRT LightTracker per feature group; validated by
                         Lucas-Kanade optical flow; refined by template matching
                         every N frames.
    Re-detection       – RedetectionManager with Kalman-predicted search region.
    Outlier filter     – Median-motion consensus; per-feature-group thresholds.
    Pose estimation    – PnPPoseEstimator: PnP with up to 6 correspondences,
                         motion-VP X-axis correction, EMA + SLERP smoothing.
    Pose freeze        – After a configurable frame index the last accepted pose
                         is held fixed so that the 3-D box does not drift.
    """

    # Frame index at which the pose (bbox, axes, origin) is frozen permanently
    FREEZE_FRAME = 195

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        config: dict,
        frame_width: int,
        frame_height: int,
        fps: int = 30,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self.config        = config
        self.frame_width   = frame_width
        self.frame_height  = frame_height
        self.fps           = fps
        self.dt            = 1.0 / fps

        detection_params = config.get('detection_params', {})

        # --- Outlier filter parameters ---
        self.prev_features_for_filter = None
        self.max_point_jump    = 25.0   # max deviation from median movement (px)
        self.freeze_eps        = 2.0    # movement below this → point is "frozen"
        self.frozen_threshold  = 3      # if >= N points frozen → bbox_is_frozen
        self.last_frozen_count = 0
        self.bbox_is_frozen    = False  # set by filter; used for display only

        # --- Pose freeze state ---
        self.pose_frozen   = False
        self.last_good_pose = None

        # --- Core components ---
        self.detector = AdvancedDetector(detection_params)

        self.trackers = {
            name: LightTracker(detection_params)
            for name in ('top', 'outer', 'bottom')
        }

        basic_detector = LightDetector(detection_params)
        selector       = CandidateSelector(detection_params, frame_width, frame_height)
        self.redetector = RedetectionManager(basic_detector, selector, detection_params)

        self.pnp_solver = PnPPoseEstimator(
            camera_matrix, config['vehicle_model'], dist_coeffs
        )
        
        self.vp_geom_solver = VPPoseEstimator(camera_matrix, config['vehicle_model'])

        cam_params = CameraParameters(camera_matrix, dist_coeffs)
        self.bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])

        # --- Tracking state ---
        self.vehicle_detected      = False
        self.last_known_features   = None
        self.last_known_plate_bottom = None
        self.tracking_failures     = 0
        self.MAX_TRACKING_FAILURES = 5

        # --- Template matching state ---
        self.keypoint_templates   = None
        self.refine_every_n_frames = 3
        self.frame_count           = 0

        # --- Optical flow state ---
        self.prev_frame_gray = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        print("VehicleTrackingPipeline initialised:")
        print(f"  Detector   : AdvancedDetector (multi-feature)")
        print(f"  Trackers   : {len(self.trackers)} feature groups")
        print(f"  Template refinement every {self.refine_every_n_frames} frames")
        print(f"  Pose freeze at frame {self.FREEZE_FRAME}")
        print(f"  TTI enabled  (dt = {self.dt:.4f} s)")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_initial_vehicle(
        self, frame: np.ndarray
    ) -> Tuple[Optional[dict], Optional[np.ndarray]]:
        """
        Run full multi-feature detection on a single frame.

        Args:
            frame: BGR input frame.

        Returns:
            (features_dict, plate_bottom) where features_dict has keys
            'top', 'outer', 'bottom' each as (2, 2) float32 arrays, or
            (None, None) if detection failed.
        """
        keypoints = self.detector.detect_all_multifeature(frame)
        if keypoints is None:
            return None, None

        self.keypoint_templates = keypoints.templates
        return keypoints.tail_lights_features, keypoints.plate_bottom

    # ------------------------------------------------------------------
    # Template matching refinement
    # ------------------------------------------------------------------

    def _extract_keypoint_template(
        self, frame: np.ndarray, center: Tuple[int, int], size: int = 25
    ) -> np.ndarray:
        """Crop a square grayscale patch centred on a keypoint."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = center
        half   = size // 2
        x1 = max(0, cx - half);  x2 = min(gray.shape[1], cx + half + 1)
        y1 = max(0, cy - half);  y2 = min(gray.shape[0], cy + half + 1)
        patch = gray[y1:y2, x1:x2].copy()
        if patch.shape[0] < size or patch.shape[1] < size:
            patch = cv2.resize(patch, (size, size))
        return patch

    def _refine_keypoint_with_template(
        self,
        frame: np.ndarray,
        approx_center: Tuple[int, int],
        template: np.ndarray,
        search_radius: int = 25,
    ) -> Tuple[int, int]:
        """
        Refine a keypoint position via normalised cross-correlation template
        matching within a local search window.
        """
        if template is None or template.size == 0:
            return approx_center

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = approx_center

        x1 = max(0, cx - search_radius);  x2 = min(w, cx + search_radius)
        y1 = max(0, cy - search_radius);  y2 = min(h, cy + search_radius)
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0 or roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return approx_center

        try:
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.55:
                rx = x1 + max_loc[0] + template.shape[1] // 2
                ry = y1 + max_loc[1] + template.shape[0] // 2
                if np.hypot(rx - cx, ry - cy) < search_radius * 1.5:
                    return (rx, ry)
        except cv2.error:
            pass

        return approx_center

    def refine_features_with_templates(
        self, frame: np.ndarray, tracker_features: dict
    ) -> dict:
        """Refine all tracked keypoints using stored template patches."""
        if self.keypoint_templates is None:
            return tracker_features

        refined = {}
        for name, centers in tracker_features.items():
            templates = self.keypoint_templates.get(name)
            if templates is None or len(templates) != 2:
                refined[name] = centers
                continue

            refined[name] = tuple(
                self._refine_keypoint_with_template(frame, c, t, search_radius=50)
                for c, t in zip(centers, templates)
            )

        return refined

    # ------------------------------------------------------------------
    # Optical flow validation
    # ------------------------------------------------------------------

    def validate_with_optical_flow(
        self, frame: np.ndarray, tracker_features: dict
    ) -> dict:
        """
        Validate tracker output against Lucas-Kanade optical flow predictions.

        If any feature point diverges from its flow prediction by more than
        20 px the flow result is used in place of the tracker result for all
        feature groups.

        Args:
            frame:            Current BGR frame.
            tracker_features: Feature dict from the CSRT trackers.

        Returns:
            Validated (or flow-corrected) feature dict.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None or self.last_known_features is None:
            self.prev_frame_gray = gray
            return tracker_features

        # Collect all previous points into a single array for batch LK tracking
        all_prev_pts = []
        for name in ('top', 'outer', 'bottom'):
            if name in self.last_known_features:
                all_prev_pts.extend(self.last_known_features[name])

        if not all_prev_pts:
            self.prev_frame_gray = gray
            return tracker_features

        prev_pts = np.array(all_prev_pts, dtype=np.float32).reshape(-1, 1, 2)

        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, gray, prev_pts, None, **self.lk_params
            )
        except cv2.error:
            self.prev_frame_gray = gray
            return tracker_features

        self.prev_frame_gray = gray

        if next_pts is None or status is None:
            return tracker_features

        # Reconstruct per-group flow predictions
        flow_features = {}
        idx = 0
        for name in ('top', 'outer', 'bottom'):
            if name not in tracker_features:
                continue
            pts = []
            for _ in range(2):
                if idx < len(next_pts) and status[idx] == 1:
                    pts.append(tuple(map(int, next_pts[idx][0])))
                else:
                    pts.append(tracker_features[name][_ if len(pts) == 0 else 1])
                idx += 1
            flow_features[name] = tuple(pts)

        # Use flow if any tracker point drifts more than 20 px from prediction
        drift_detected = any(
            np.any(
                np.linalg.norm(
                    np.array(tracker_features[n], dtype=np.float32)
                    - np.array(flow_features[n],   dtype=np.float32),
                    axis=1
                ) > 20.0
            )
            for n in tracker_features if n in flow_features
        )

        if drift_detected:
            print(f"  Tracker drift detected – switching to optical flow.")
            return flow_features

        return tracker_features

    # ------------------------------------------------------------------
    # Tracking + re-detection
    # ------------------------------------------------------------------

    def track_or_redetect(self, frame: np.ndarray) -> Optional[dict]:
        """
        Attempt to track the vehicle using the CSRT trackers, with optical
        flow validation, template refinement, and automatic re-detection on
        repeated failure.

        Also updates the plate bottom estimate whenever tracking succeeds.

        Returns:
            Feature dict or None if tracking is lost beyond recovery.
        """
        # Layer 1: CSRT trackers
        tracker_features = {}
        all_ok = True

        for name, tracker in self.trackers.items():
            if not tracker.is_initialized:
                all_ok = False
                break
            ok, centers = tracker.update(frame)
            if ok:
                tracker_features[name] = centers
            else:
                all_ok = False
                break

        if all_ok and tracker_features:
            # Layer 2: optical flow validation
            features = self.validate_with_optical_flow(frame, tracker_features)

            # Layer 3: periodic template refinement
            self.frame_count += 1
            if self.frame_count % self.refine_every_n_frames == 0:
                refined = self.refine_features_with_templates(frame, features)
                if refined != features:
                    # Update stored templates to track appearance changes
                    for name, centers in refined.items():
                        if name in self.keypoint_templates:
                            for i, center in enumerate(centers):
                                tmpl = self._extract_keypoint_template(frame, center)
                                if tmpl.size > 0:
                                    self.keypoint_templates[name][i] = tmpl
                features = refined

            # Update plate bottom while tracking is healthy
            plate_bottom = self.detector.update_plate_bottom_only(frame, features)
            if plate_bottom is not None:
                self.last_known_plate_bottom = plate_bottom

            self.last_known_features = features
            self.tracking_failures   = 0

            if 'outer' in features:
                self.redetector.update_kalman(features['outer'])

            return features

        # Layer 4: re-detection after repeated failures
        self.tracking_failures += 1

        if self.tracking_failures >= 3:
            last_outer = (self.last_known_features.get('outer')
                          if self.last_known_features else None)
            new_centers = self.redetector.redetect(
                frame, last_known_centers=last_outer, search_region_scale=2.5
            )

            if new_centers is not None:
                features_dict, plate_bottom = self.detect_initial_vehicle(frame)
                if features_dict is not None:
                    for name, centers in features_dict.items():
                        if name in self.trackers:
                            if isinstance(centers, np.ndarray):
                                centers = tuple(tuple(map(int, pt)) for pt in centers)
                            self.trackers[name].reinitialize(frame, centers)

                    self.last_known_features     = features_dict
                    self.last_known_plate_bottom = plate_bottom
                    self.tracking_failures       = 0
                    return features_dict

        # Fallback: if fewer than 40 % of expected points remain, force reset
        if self.last_known_features:
            n_pts = sum(len(v) for v in self.last_known_features.values())
            if n_pts < 6 * 0.4:
                print(f"  Critical feature loss ({n_pts}/6) – resetting pipeline.")
                self.reset()
                return None

        if self.tracking_failures >= self.MAX_TRACKING_FAILURES:
            self.reset()
            return None

        return self.last_known_features

    def reset(self):
        """Reset all pipeline state (called when the vehicle is lost)."""
        self.vehicle_detected        = False
        self.last_known_features     = None
        self.last_known_plate_bottom = None
        self.tracking_failures       = 0
        self.pose_frozen             = False
        self.last_good_pose          = None
        self.bbox_is_frozen          = False

        for tracker in self.trackers.values():
            tracker.reset()

        self.keypoint_templates       = None
        self.frame_count              = 0
        self.prev_frame_gray          = None
        self.prev_features_for_filter = None
        self.last_frozen_count        = 0

        self.pnp_solver.reset_tti_history()
        self.pnp_solver.reset_temporal_smoothing()
        self.pnp_solver.reset_vp_persistence()

    # ------------------------------------------------------------------
    # Outlier filter
    # ------------------------------------------------------------------

    def filter_outlier_points(self, current_features: dict) -> dict:
        """
        Suppress spurious single-point jumps using a median-motion consensus.

        For each frame, the movement vector of every tracked point is computed.
        The component-wise median across all points gives the consensus motion.
        Points that deviate from the consensus by more than a per-group threshold
        are replaced by  prev + median_motion  (a kinematic prediction).

        The method also counts how many points are near-stationary and sets
        ``self.bbox_is_frozen`` and ``self.last_frozen_count`` accordingly.

        Args:
            current_features: Feature dict from the tracker/flow step.

        Returns:
            Filtered feature dict with outlier points replaced.
        """
        if self.prev_features_for_filter is None:
            self.prev_features_for_filter = {
                k: np.array(v, dtype=np.float32) for k, v in current_features.items()
            }
            self.last_frozen_count = 0
            self.bbox_is_frozen    = False
            return current_features

        current = {
            k: np.array(v, dtype=np.float32) for k, v in current_features.items()
        }

        # Collect all movement vectors
        all_movements = []
        for name in ('top', 'outer', 'bottom'):
            if name not in current or name not in self.prev_features_for_filter:
                continue
            curr = current[name]
            prev = self.prev_features_for_filter[name]
            for i in range(min(len(curr), len(prev))):
                all_movements.append(curr[i] - prev[i])

        if len(all_movements) < 3:
            self.last_frozen_count = 0
            self.bbox_is_frozen    = False
            return current

        movements       = np.array(all_movements, dtype=np.float32)
        median_movement = np.median(movements, axis=0)

        # Count near-stationary points
        frozen_count = int(np.sum(np.linalg.norm(movements, axis=1) < self.freeze_eps))
        self.last_frozen_count = frozen_count

        prev_frozen = self.bbox_is_frozen
        self.bbox_is_frozen = (frozen_count >= self.frozen_threshold)
        if self.bbox_is_frozen and not prev_frozen:
            print(f"  Outlier filter: {frozen_count}/{len(movements)} points static – bbox frozen.")
        elif not self.bbox_is_frozen and prev_frozen:
            print(f"  Outlier filter: {frozen_count}/{len(movements)} points static – bbox unfrozen.")

        # Replace outlier points with median-based prediction
        # Per-group thresholds reflect how reliable each sub-point type is:
        #   outer  – tightest (lateral edge, most stable)
        #   top    – nominal
        #   bottom – most permissive (reflections cause occasional jumps)
        group_threshold = {
            'outer':  self.max_point_jump * 0.8,
            'top':    self.max_point_jump,
            'bottom': self.max_point_jump * 1.2,
        }

        filtered = {}
        for name in ('top', 'outer', 'bottom'):
            if name not in current:
                continue
            if name not in self.prev_features_for_filter:
                filtered[name] = current[name]
                continue

            curr = current[name]
            prev = self.prev_features_for_filter[name]
            thr  = group_threshold.get(name, self.max_point_jump)
            pts  = []

            for i in range(len(curr)):
                if i >= len(prev):
                    pts.append(curr[i])
                    continue
                deviation = np.linalg.norm((curr[i] - prev[i]) - median_movement)
                if deviation > thr:
                    pts.append((prev[i] + median_movement).astype(np.float32))
                else:
                    pts.append(curr[i])

            filtered[name] = np.array(pts, dtype=np.float32)

        self.prev_features_for_filter = {k: v.copy() for k, v in filtered.items()}
        return filtered

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        Process a single tracking frame end-to-end.

        Workflow
        --------
        1. Initial detection (first call) or tracking (subsequent calls).
        2. Outlier filtering on the tracked features.
        3. Pose freeze check: at frame FREEZE_FRAME the last accepted pose is
           locked and ``pose_for_bbox`` will always return that locked pose.
        4. PnP pose estimation (skipped if frozen or features are missing).
        5. Update of ``last_good_pose`` (only while not frozen).

        The result dict distinguishes between:
            'pose'         – the pose estimated for this frame (may be None).
            'pose_for_bbox' – the pose used for 3-D rendering; equals the
                             current pose when not frozen, or ``last_good_pose``
                             when frozen.

        Args:
            frame:     BGR input frame.
            frame_idx: Current frame index (0-based).

        Returns:
            Result dict with keys: 'success', 'features', 'plate_bottom',
            'pose', 'pose_for_bbox', 'bbox_frozen', 'motion_type', 'status'.
        """
        result = {
            'success':      False,
            'features':     None,
            'plate_bottom': None,
            'pose':         None,
            'pose_for_bbox': None,
            'bbox_frozen':  False,
            'motion_type':  'UNKNOWN',
            'status':       'processing',
        }

        # --- Detection or tracking ---
        if not self.vehicle_detected:
            features_dict, plate_bottom = self.detect_initial_vehicle(frame)
            if features_dict is not None:
                for name, centers in features_dict.items():
                    if name in self.trackers:
                        if isinstance(centers, np.ndarray):
                            centers = tuple(tuple(map(int, pt)) for pt in centers)
                        self.trackers[name].initialize(frame, centers)

                self.vehicle_detected        = True
                self.last_known_features     = features_dict
                self.last_known_plate_bottom = plate_bottom

                if 'outer' in features_dict:
                    self.redetector.update_kalman(features_dict['outer'])

                result.update(success=True, features=features_dict,
                               plate_bottom=plate_bottom, status='initial_detection')
        else:
            features = self.track_or_redetect(frame)
            if features is not None:
                result.update(success=True, features=features,
                               plate_bottom=self.last_known_plate_bottom, status='tracking')
            else:
                result['status'] = 'lost'

        # --- Outlier filter ---
        if result['success'] and result['features'] is not None:
            result['features'] = self.filter_outlier_points(result['features'])

        # --- Pose freeze check ---
        if frame_idx >= self.FREEZE_FRAME:
            self.pose_frozen = True
        result['bbox_frozen'] = self.pose_frozen

        # --- Pose estimation ---
        if result['success'] and result['features'] is not None:
            try:
                features_t2 = {
                    name: np.array(result['features'][name], dtype=np.float32)
                    for name in ('outer', 'top', 'bottom')
                    if name in result['features']
                }

                if len(features_t2) < 3:
                    result['pose'] = None
                else:
                    plate_bottom_t2 = None
                    if result['plate_bottom'] is not None:
                        pb = np.array(result['plate_bottom'], dtype=np.float32)
                        if np.linalg.norm(pb[1] - pb[0]) > 10.0:
                            plate_bottom_t2 = pb

                    pose_data = self.pnp_solver.estimate_pose_multifeature(
                        features_t2, plate_bottom_t2, frame_idx
                    )

                    if pose_data is not None:
                        result['pose']         = pose_data
                        result['motion_type']  = pose_data.get('motion_type', 'UNKNOWN')
                        if not self.pose_frozen:
                            self.last_good_pose = pose_data
                    else:
                        result['pose'] = None

            except Exception as e:
                import traceback
                print(f"  Frame {frame_idx}: pose estimation error – {e}")
                traceback.print_exc()
                
                
                
        if result['pose'] is not None:
            # Stima anche con VP geometrico
            vp_result = self.vp_geom_solver.estimate_pose_vp(
                result['features'], frame_idx
            )
            if vp_result:
                result['vp_estimate'] = vp_result        
                        

        # --- Assign rendering pose ---
        # Use the current pose while not frozen; fall back to last_good_pose when frozen.
        if result['pose'] is not None and not self.pose_frozen:
            result['pose_for_bbox'] = result['pose']
        else:
            result['pose_for_bbox'] = self.last_good_pose

        return result



def save_reprojection_error_plot(
    reproj_data: list,
    freeze_frame: int,
    output_path: str,
    video_name: str,
) -> None:
    """
    Generate and save a reprojection error vs frame index plot.

    Args:
        reproj_data:  List of (frame_idx, mean_reproj_error_px) tuples.
        freeze_frame: Frame index at which the pose is frozen (drawn as vertical line).
        output_path:  Full path for the output PNG file.
        video_name:   Video name used in the plot title.
    """
    if not reproj_data:
        print("  No reprojection error data to plot.")
        return

    frames = [d[0] for d in reproj_data]
    errors = [d[1] for d in reproj_data]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(frames, errors, color='steelblue', linewidth=1.0, label='Mean reprojection error')

    # Running average (window = 10 frames) for visual clarity
    if len(errors) >= 10:
        kernel = np.ones(10) / 10
        smooth = np.convolve(errors, kernel, mode='valid')
        offset = 10 // 2
        ax.plot(
            frames[offset: offset + len(smooth)],
            smooth,
            color='tomato', linewidth=2.0, linestyle='--',
            label='10-frame moving average'
        )

    # Pose freeze vertical line
    ax.axvline(x=freeze_frame, color='orange', linewidth=1.5,
               linestyle=':', label=f'Pose freeze (frame {freeze_frame})')

    # Shaded region after freeze
    ax.axvspan(freeze_frame, max(frames), alpha=0.08, color='orange')

    ax.set_xlabel('Frame index', fontsize=11)
    ax.set_ylabel('Mean reprojection error (px)', fontsize=11)
    ax.set_title(f'Per-frame reprojection error — {video_name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  Reprojection error plot saved to: {output_path}")
    

# ===========================================================================
# Video processing
# ===========================================================================

def process_video(
    video_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    config: dict,
) -> bool:
    """
    Run the full tracking and pose-estimation pipeline on a video file.

    Output files written to  data/videos/output/:
        {stem}_output.avi       – annotated video with keypoints, bbox, pose info
        {stem}_debug_mask.avi   – debug visualisation (mask + reprojection)

    Per-frame pose data saved as .npz to  data/results/vanishing_point/.

    Args:
        video_path:    Path to the input video.
        camera_matrix: 3x3 camera intrinsic matrix.
        dist_coeffs:   Distortion coefficients.
        config:        Loaded configuration dict.

    Returns:
        True on successful completion, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return False

    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem         = Path(video_path).stem

    print(f"\nVideo : {Path(video_path).name}")
    print(f"       {width}x{height} @ {fps} fps  ({total_frames} frames)")

    # Output paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{stem}_output.avi"
    debug_path  = OUTPUT_DIR / f"{stem}_debug_mask.avi"

    writer       = VideoWriter(str(output_path), fps, (width, height), codec='MJPG')
    writer_debug = VideoWriter(str(debug_path),  fps, (width, height), codec='MJPG')

    # Results directory (folder name preserved as-is)
    results_dir = project_root / "data" / "results" / "vanishing_point"
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline = VehicleTrackingPipeline(
        camera_matrix, dist_coeffs, config, width, height, fps
    )

    # Require this many consecutive successful detections before initialising
    # the trackers, to avoid locking on to a transient false positive.
    REQUIRED_CONSECUTIVE = 5
    consecutive          = 0
    vehicle_initialized  = False

    frame_idx        = 0
    reproj_errors = []    # list of (frame_idx, float) -- collected during loop
    prev_features    = None
    prev_plate_bottom = None

    print(f"\nProcessing (requires {REQUIRED_CONSECUTIVE} consecutive detections to start)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp       = frame.copy()
        debug_mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # ---------------------------------------------------------------
        # Phase 1: initial detection stability check
        # ---------------------------------------------------------------
        if not vehicle_initialized:
            features_dict, plate_bottom = pipeline.detect_initial_vehicle(frame)

            if features_dict is not None:
                consecutive += 1

                if consecutive >= REQUIRED_CONSECUTIVE:
                    for name, centers in features_dict.items():
                        if name in pipeline.trackers:
                            if isinstance(centers, np.ndarray):
                                centers = tuple(tuple(map(int, pt)) for pt in centers)
                            pipeline.trackers[name].initialize(frame, centers)

                    pipeline.vehicle_detected        = True
                    pipeline.last_known_features     = features_dict
                    pipeline.last_known_plate_bottom = plate_bottom

                    if 'outer' in features_dict:
                        pipeline.redetector.update_kalman(features_dict['outer'])

                    vehicle_initialized  = True
                    prev_features        = features_dict
                    prev_plate_bottom    = plate_bottom

                    print(f"  Tracker initialised at frame {frame_idx}.")

                    for name, centers in features_dict.items():
                        color = (0, 255, 0) if name == 'outer' else (0, 200, 200)
                        for pt in centers:
                            cv2.circle(frame_disp, tuple(map(int, pt)), 6, color, -1)
                            cv2.circle(frame_disp, tuple(map(int, pt)), 8, (255, 255, 255), 2)

                    cv2.putText(frame_disp, "Tracking initialised",
                                (width // 2 - 130, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    remaining = REQUIRED_CONSECUTIVE - consecutive
                    cv2.putText(frame_disp,
                                f"Vehicle detected – stabilising ({remaining} more needed)",
                                (width // 2 - 320, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
            else:
                if consecutive > 0:
                    print(f"  Frame {frame_idx}: detection lost during stabilisation.")
                consecutive = 0
                cv2.putText(frame_disp, "Waiting for vehicle...",
                            (width // 2 - 180, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            writer.write(frame_disp)
            frame_idx += 1
            continue

        # ---------------------------------------------------------------
        # Phase 2: normal tracking
        # ---------------------------------------------------------------
        result = pipeline.process_frame(frame, frame_idx)

        # --- Debug mask ---
        if result['success'] and result['features'] is not None:
            ref_pose = result.get('pose') or result.get('pose_for_bbox')
            rvec_dbg = ref_pose['rvec'] if ref_pose else None
            tvec_dbg = ref_pose['tvec'] if ref_pose else None

            debug_mask_frame = DrawUtils.create_debug_mask_frame(
                frame,
                pipeline.detector,
                result['features'],
                result['plate_bottom'],
                rvec=rvec_dbg,
                tvec=tvec_dbg,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                frame_idx=frame_idx,
            )
        else:
            msg = ("TRACKING LOST" if result['status'] == 'lost'
                   else "Waiting for vehicle...")
            colour = (0, 0, 255) if result['status'] == 'lost' else (255, 255, 255)
            cv2.putText(debug_mask_frame, msg,
                        (width // 2 - 150, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        # --- Overlay: status text ---
        if result['status'] == 'lost':
            cv2.putText(frame_disp, "LOST – searching...",
                        (width // 2 - 130, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        elif result['success']:
            features    = result['features']
            pose        = result.get('pose')
            pose_for_bbox = result.get('pose_for_bbox') or pose
            bbox_frozen = result.get('bbox_frozen', False) or pipeline.pose_frozen

            # --- Keypoints ---
            if features:
                colour_map = {'outer': (0, 255, 0), 'top': (255, 255, 0), 'bottom': (0, 255, 255)}
                for name, pts in features.items():
                    colour = colour_map.get(name, (200, 200, 200))
                    for pt in pts:
                        cv2.circle(frame_disp, tuple(map(int, pt)), 5, colour, -1)

            # --- Plate bottom ---
            if result['plate_bottom'] is not None:
                pb = result['plate_bottom']
                for pt in pb:
                    cv2.circle(frame_disp, tuple(map(int, pt)), 5, (255, 0, 255), -1)
                cv2.line(frame_disp, tuple(map(int, pb[0])), tuple(map(int, pb[1])),
                         (255, 0, 255), 3, cv2.LINE_AA)

            # --- Plate corners (from detector internal state) ---
            if pipeline.detector.prev_plate_corners is not None:
                for key in ('BL', 'BR'):
                    pt = pipeline.detector.prev_plate_corners[key]
                    cv2.circle(frame_disp, pt, 5, (255, 128, 255), -1)

            # --- 3-D rendering (bbox, axes, origin) ---
            if pose_for_bbox is not None:
                rvec_b = pose_for_bbox['rvec']
                tvec_b = pose_for_bbox['tvec']

                draw_3d_axes(frame_disp, rvec_b, tvec_b, camera_matrix, dist_coeffs, 1.5, 4)

                bbox_2d = pipeline.bbox_projector.project_bbox(rvec_b, tvec_b)
                if bbox_2d is not None:
                    box_colour = (0, 165, 255) if bbox_frozen else (0, 255, 0)
                    draw_bbox_3d(frame_disp, bbox_2d, color=box_colour, thickness=2)
                    if bbox_frozen:
                        cv2.putText(frame_disp, "BBOX FROZEN",
                                    (width // 2 - 90, height - 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                # Origin marker and forward-direction arrow
                origin_3d  = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                forward_3d = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
                origin_2d,  _ = cv2.projectPoints(origin_3d,  rvec_b, tvec_b, camera_matrix, dist_coeffs)
                forward_2d, _ = cv2.projectPoints(forward_3d, rvec_b, tvec_b, camera_matrix, dist_coeffs)
                o_px = tuple(map(int, origin_2d[0][0]))
                f_px = tuple(map(int, forward_2d[0][0]))
                cv2.drawMarker(frame_disp, o_px, (0, 165, 255), cv2.MARKER_CROSS, 40, 5)
                cv2.circle(frame_disp, o_px, 20, (0, 165, 255), 3)
                cv2.arrowedLine(frame_disp, o_px, f_px, (255, 255, 0), 4, tipLength=0.3)

                # Yaw text
                R_ref    = (pose or pose_for_bbox)['R']
                yaw_val  = (pipeline.pnp_solver.yaw_smooth
                            if pipeline.pnp_solver.yaw_smooth is not None
                            else pipeline.pnp_solver.extract_yaw_from_rotation(R_ref))
                cv2.putText(frame_disp, f"Yaw: {np.degrees(yaw_val):+6.1f} deg",
                            (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Distance and TTI
                ref_pose_for_info = pose if pose else pose_for_bbox
                distance = float(np.linalg.norm(ref_pose_for_info['tvec']))
                draw_tracking_info(frame_disp, frame_idx, "PnP Estimation", distance, 2)

                if pose and pose.get('tti') is not None:
                    tti       = pose['tti']
                    tti_valid = pose.get('tti_valid', True)
                    tti_color = (0, 255, 0) if tti_valid else (0, 0, 255)
                    tti_str   = (f"TTI: {tti:.1f} s" if abs(tti) < 100 else "TTI: >100 s")
                    tti_str  += " (moving away)" if tti < 0 else " (approaching)"
                    cv2.putText(frame_disp, tti_str,
                                (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tti_color, 2)

                # --- Save per-frame pose data ---
                if pose:
                    save_data = {
                        'rvec':         pose['rvec'],
                        'tvec':         pose['tvec'],
                        'R':            pose['R'],
                        'method':       'pnp_multifeature',
                        'motion_type':  result['motion_type'],
                        'tracking_failures': pipeline.tracking_failures,
                        'tti':          pose.get('tti'),
                        'tti_valid':    pose.get('tti_valid'),
                        'pose_method':  pose.get('debug', {}).get('method', 'unknown'),
                        'bbox_frozen':  bbox_frozen,
                        'frozen_count': pipeline.last_frozen_count,
                        'distance_pnp': float(np.linalg.norm(pose['tvec'])),
                        'reproj_error': pose.get('reproj_error'),
                    }
                    
                    # Aggiungi dati VP se disponibili
                    if 'vp_estimate' in result:
                        save_data['distance_vp'] = result['vp_estimate']['distance_vp']
                        save_data['vx'] = result['vp_estimate'].get('vx')
                                    
                    
                    # Collect reprojection error for the plot
                    if pose and pose.get('reproj_error') is not None:
                        reproj_errors.append((frame_idx, float(pose['reproj_error'])))


                    for feat_name in ('top', 'outer', 'bottom'):
                        if prev_features and feat_name in prev_features:
                            save_data[f'lights_{feat_name}_frame1'] = (
                                np.array(prev_features[feat_name])
                            )
                        if features and feat_name in features:
                            save_data[f'lights_{feat_name}_frame2'] = (
                                np.array(features[feat_name])
                            )

                    if prev_plate_bottom is not None:
                        save_data['plate_bottom_frame1'] = np.array(prev_plate_bottom)
                    if result['plate_bottom'] is not None:
                        save_data['plate_bottom_frame2'] = np.array(result['plate_bottom'])

                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(str(pose_file), **save_data)

            # --- Vanishing-point convergence lines ---
            if pose and 'vx' in pose and 'vy' in pose:
                features_for_vp = {
                    k: np.array(features[k], dtype=np.float32)
                    for k in ('outer', 'top', 'bottom') if k in features
                }
                if result['plate_bottom'] is not None:
                    features_for_vp['plate_bottom'] = np.array(
                        result['plate_bottom'], dtype=np.float32
                    )
                prev_vp = {
                    k: np.array(prev_features[k], dtype=np.float32)
                    for k in ('outer', 'top') if prev_features and k in prev_features
                }
                DrawUtils.draw_vp_convergence(
                    frame_disp,
                    features_curr=features_for_vp,
                    features_prev=prev_vp if prev_vp else None,
                    vx_motion=pose.get('vx'),
                    vy_lateral=pose.get('vy'),
                    show_labels=True,
                )

            # --- Status / failure counter ---
            status_color = (0, 255, 0) if pipeline.tracking_failures == 0 else (0, 200, 200)
            cv2.putText(frame_disp, f"Status: {result['status']}",
                        (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            if pipeline.tracking_failures > 0:
                cv2.putText(frame_disp,
                            f"Failures: {pipeline.tracking_failures}/{pipeline.MAX_TRACKING_FAILURES}",
                            (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

        draw_motion_type_overlay(frame_disp, result['motion_type'])

        writer.write(frame_disp)
        writer_debug.write(debug_mask_frame)

        # Update previous-frame references
        if result['success'] and result['features'] is not None:
            prev_features = result['features']
            if result['plate_bottom'] is not None:
                prev_plate_bottom = result['plate_bottom']

        # Progress report every 30 frames
        if frame_idx % 30 == 0:
            frozen_tag = f" [frozen: {pipeline.last_frozen_count} pts]" if pipeline.bbox_is_frozen else ""
            print(f"  Frame {frame_idx:4d}/{total_frames}"
                  f"  ({frame_idx / total_frames * 100:5.1f} %)"
                  f"  {result['status']}{frozen_tag}")

        frame_idx += 1

    cap.release()
    writer.release()
    writer_debug.release()

    # --- Generate reprojection error plot ---
    plot_path = OUTPUT_DIR / f"{stem}_reproj_error.png"
    save_reprojection_error_plot(
        reproj_data=reproj_errors,
        freeze_frame=VehicleTrackingPipeline.FREEZE_FRAME,
        output_path=str(plot_path),
        video_name=Path(video_path).name,
    )

    print(f"\nProcessing complete.")
    print(f"  Annotated video : {output_path}")
    print(f"  Debug mask      : {debug_path}")
    print(f"  Pose results    : {results_dir}")
    print(f"  Reprojection error plot : {plot_path}")

    return True


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    """Interactive menu: recalibrate camera or process a video."""
    print("\n" + "=" * 70)
    print(" VEHICLE TAIL-LIGHT TRACKING SYSTEM")
    print("=" * 70)

    try:
        config = load_all_configs()
        calib_file = config['camera_config']['camera']['calibration_file']
        camera_matrix, dist_coeffs = load_camera_calibration(calib_file)
        print("Configuration loaded.")
        print(f"  Camera calibration: {calib_file}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    while True:
        print("\n" + "-" * 70)
        print("  1 - Recalibrate camera")
        print("  2 - Process video")
        print("  0 - Exit")
        print("-" * 70)

        try:
            choice = input("Select: ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if choice == "0":
            print("Bye.")
            break

        elif choice == "1":
            result = recalibrate_camera(config)
            if result is not None:
                camera_matrix, dist_coeffs = result
                print("Camera parameters updated for this session.")

        elif choice == "2":
            video_path, _ = choose_video()
            if video_path is None:
                continue
            try:
                process_video(video_path, camera_matrix, dist_coeffs, config)
            except Exception as e:
                import traceback
                print(f"\nError during processing: {e}")
                traceback.print_exc()

        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()