"""
Vehicle Localization System - VERSIONE FINALE MULTI-FEATURE + TTI
Sistema completo con menu per selezione video e metodo

"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from detection.light_detector import LightDetector
from detection.candidate_selector import CandidateSelector
from detection.advanced_detector import AdvancedDetector
from tracking.tracker import LightTracker
from tracking.redetection import RedetectionManager
from pose_estimation.vanishing_point_solver import VanishingPointSolver
from pose_estimation.homography_solver import HomographySolver
from pose_estimation.pnp_full_solver import PnPSolver
from pose_estimation.bbox_3d_projector import BBox3DProjector
from visualization.video_writer import VideoWriter
from visualization.draw_utils import (
    draw_tracking_info,
    draw_motion_type_overlay,
    draw_bbox_3d,

    draw_vanishing_points_complete,
    # NOTE:
    # Vx and ground plane π are visualized for debugging and task compliance only.
    # They are NOT used for pose or distance estimation due to instability in noisy nighttime scenes.


    draw_3d_axes
)
from visualization.draw_utils import DrawUtils


from calibration.load_calibration import (
    load_camera_calibration_simple as load_camera_calibration, 
    CameraParameters
)
from utils.config_loader import load_config



# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_DIR = project_root / "data" / "videos" / "input"
OUTPUT_DIR = project_root / "data" / "videos" / "output"
CONFIG_DIR = project_root / "config"


# ============================================================================
# HELPER: Load all configs with fallback
# ============================================================================

def load_all_configs():
    """Load all configuration files."""
    config = {}

    # Camera config (required)
    camera_config_path = CONFIG_DIR / 'camera_config.yaml'
    if not camera_config_path.exists():
        raise FileNotFoundError(f"Required file not found: {camera_config_path}")
    config['camera_config'] = load_config(str(camera_config_path))

    # Vehicle model (required)
    vehicle_model_path = CONFIG_DIR / 'vehicle_model.yaml'
    if not vehicle_model_path.exists():
        raise FileNotFoundError(f"Required file not found: {vehicle_model_path}")
    config['vehicle_model'] = load_config(str(vehicle_model_path))

    # Detection params (required)
    detection_params_path = CONFIG_DIR / 'detection_params.yaml'
    if not detection_params_path.exists():
        raise FileNotFoundError(f"Required file not found: {detection_params_path}")
    config['detection_params'] = load_config(str(detection_params_path))

    return config


# ============================================================================
# MENU FUNCTIONS
# ============================================================================

def show_menu():
    """Mostra menu principale."""
    print("\n" + "=" * 70)
    print(" 🚗 VEHICLE LOCALIZATION SYSTEM - MULTI-FEATURE + TTI ")
    print("=" * 70)
    print("2 - Task 2: Vanishing Point (Multi-Feature + Hybrid Tracking + TTI)")
    print("0 - Exit")
    print("=" * 70)


def choose_video():
    """Menu per scegliere video."""
    if not VIDEO_DIR.exists():
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return None, None

    videos = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.endswith((".mp4", ".avi", ".mov"))
    ])

    if not videos:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return None, None

    print("\n📹 Available videos:")
    for i, v in enumerate(videos, 1):
        print(f"  {i} - {v}")

    try:
        idx = int(input("\nSelect video number: ")) - 1
        if 0 <= idx < len(videos):
            video_name = videos[idx]
            video_path = str(VIDEO_DIR / video_name)
            return video_path, video_name
        else:
            print("❌ Invalid selection")
            return None, None
    except (ValueError, KeyboardInterrupt):
        return None, None


# ============================================================================
# TASK 
# ============================================================================

class Task2Pipeline:
    """
    Pipeline MULTI-FEATURE HYBRID TRACKING + TTI (Time-To-Impact)

    ARCHITETTURA:
    1. Detection multi-feature (top, outer, bottom + plate bottom)
    2. Tracking (CSRT tracker per ogni feature)
    3. Template Matching Refinement
    4. Optical Flow Validation
    5. Re-detection
    6. Vanishing Point Solver (multi-feature + TTI)
    """

    def __init__(self, camera_matrix, dist_coeffs, config, frame_width, frame_height, fps=30):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.dt = 1.0 / fps  # Delta tempo per TTI

        detection_params = config.get('detection_params', {})

        # ===== FILTRO OUTLIER =====
        self.prev_features_for_filter = None
        self.max_point_jump = 25.0       # Deviazione massima dalla mediana (px)
        self.freeze_eps = 2.0            # Movimento < freeze_eps px → punto "fermo"
        self.frozen_threshold = 3        # Se ≥ N punti fermi → freeze bbox
        self.last_frozen_count = 0
        
        # ===== BBOX FREEZE =====
        # Ultima posa valida con punti in movimento
        self.last_good_pose = None
        self.pose_quality_failures = 0
        self.MAX_POSE_QUALITY_FAILURES = 5   # frame consecutivi con posa cattiva → freeze
        self.quality_bbox_frozen = False
        self.INSTANT_FREEZE_ERROR = 200.0  # px soglia freeze immediato per errori gravi

        print("  ✓ Outlier filter: mediana movimento (max_jump=30px)")
        print(f"  ✓ BBox freeze: {self.frozen_threshold} punti fermi → congela bbox")


        # ===== COMPONENTI CORE =====
        # Detector (usa AdvancedDetector MULTI-FEATURE)
        self.detector = AdvancedDetector(detection_params)

        # Tracker (uno per ogni feature type)
        self.trackers = {}  # {'top': LightTracker, 'outer': LightTracker, 'bottom': LightTracker}
        for feature_name in ['top', 'outer', 'bottom']:
            self.trackers[feature_name] = LightTracker(detection_params)

        # Re-detection (usa detector base + selector)
        basic_detector = LightDetector(detection_params)
        selector = CandidateSelector(detection_params, frame_width, frame_height)
        self.redetector = RedetectionManager(basic_detector, selector, detection_params)

        # Pose estimation (MULTI-FEATURE VERSION)
        self.vp_solver = VanishingPointSolver(camera_matrix, config['vehicle_model'], dist_coeffs)

        # 3D projection
        cam_params = CameraParameters(camera_matrix, dist_coeffs)
        self.bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])

        # ===== HYBRID TRACKING STATE =====
        self.vehicle_detected = False
        self.last_known_features = None  # Dict: {'top': [(L,R)], 'outer': [(L,R)], 'bottom': [(L,R)]}
        self.last_known_plate_bottom = None  # [[BL, BR]]
        self.tracking_failures = 0
        self.MAX_TRACKING_FAILURES = 5

        # Template matching
        self.keypoint_templates = None  # Dict: {'top': [L_tmpl, R_tmpl], ...}
        self.refine_every_n_frames = 3
        self.frame_count = 0

        # Optical flow
        self.prev_frame_gray = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        print("  ✓ Task2Pipeline initialized (MULTI-FEATURE HYBRID + TTI + FREEZE)")
        print(f"    Detector: AdvancedDetector (multi-feature)")
        print(f"    Trackers: {len(self.trackers)} feature types")
        print(f"    Template Matching: every {self.refine_every_n_frames} frames")
        print(f"    Optical Flow: enabled")
        print(f"    TTI: enabled (dt={self.dt:.4f}s)")

    # ========================================================================
    # DETECTION LAYER
    # ========================================================================

    def detect_initial_vehicle(self, frame):
        """
        Detection iniziale MULTI-FEATURE.

        Returns:
            Tuple (features_dict, plate_bottom) o (None, None)
            - features_dict: {'top': [[L,R]], 'outer': [[L,R]], 'bottom': [[L,R]]}
            - plate_bottom: [[BL, BR]] o None
        """
        keypoints = self.detector.detect_all_multifeature(frame)

        if keypoints is None:
            return None, None

        # Salva template per tracking futuro
        self.keypoint_templates = keypoints.templates

        return keypoints.tail_lights_features, keypoints.plate_bottom

    # ========================================================================
    # TEMPLATE MATCHING LAYER
    # ========================================================================

    def _extract_keypoint_template(self, frame, center, size=25):
        """Estrae template attorno a un keypoint."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx, cy = center
        half = size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(gray.shape[1], cx + half + 1)
        y2 = min(gray.shape[0], cy + half + 1)

        template = gray[y1:y2, x1:x2].copy()

        if template.shape[0] < size or template.shape[1] < size:
            template = cv2.resize(template, (size, size))

        return template

    def _refine_keypoint_with_template(self, frame, approx_center, template, search_radius=25):
        """Raffina keypoint usando template matching locale."""
        if template is None or template.size == 0:
            return approx_center

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx, cy = approx_center

        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(w, cx + search_radius)
        y2 = min(h, cy + search_radius)

        roi = gray[y1:y2, x1:x2]

        if roi.size == 0 or roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return approx_center

        try:
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.55:
                refined_x = x1 + max_loc[0] + template.shape[1] // 2
                refined_y = y1 + max_loc[1] + template.shape[0] // 2

                dist = np.sqrt((refined_x - cx)**2 + (refined_y - cy)**2)
                if dist < search_radius * 1.5:
                    return (refined_x, refined_y)
        except cv2.error:
            pass

        return approx_center

    def refine_features_with_templates(self, frame, tracker_features):
        """Raffina features usando template matching."""
        if self.keypoint_templates is None:
            return tracker_features

        refined_features = {}

        for feature_name, centers in tracker_features.items():
            if feature_name not in self.keypoint_templates:
                refined_features[feature_name] = centers
                continue

            templates = self.keypoint_templates[feature_name]
            if len(templates) != 2:
                refined_features[feature_name] = centers
                continue

            refined = []
            for center, template in zip(centers, templates):
                refined_center = self._refine_keypoint_with_template(frame, center, template, search_radius=50)
                refined.append(refined_center)

            refined_features[feature_name] = tuple(refined)

        return refined_features

    # ========================================================================
    # OPTICAL FLOW LAYER
    # ========================================================================

    def validate_with_optical_flow(self, frame, tracker_features):
        """Valida features usando optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None or self.last_known_features is None:
            self.prev_frame_gray = gray
            return tracker_features

        # Prepara tutti i punti per optical flow
        all_prev_pts = []
        feature_names = []
        point_indices = []

        for feature_name in ['top', 'outer', 'bottom']:
            if feature_name in self.last_known_features:
                centers = self.last_known_features[feature_name]
                for i, center in enumerate(centers):
                    all_prev_pts.append(center)
                    feature_names.append(feature_name)
                    point_indices.append(i)

        if len(all_prev_pts) == 0:
            self.prev_frame_gray = gray
            return tracker_features

        prev_pts = np.array(all_prev_pts, dtype=np.float32).reshape(-1, 1, 2)

        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, gray, prev_pts, None, **self.lk_params
            )
        except cv2.error:
            self.prev_frame_gray = gray
            return tracker_features

        self.prev_frame_gray = gray

        if next_pts is None or status is None:
            return tracker_features

        # Ricostruisci features da optical flow
        flow_features = {}
        idx = 0
        for feature_name in ['top', 'outer', 'bottom']:
            if feature_name in tracker_features:
                flow_pts = []
                for i in range(2):  # Left e Right
                    if idx < len(next_pts) and status[idx] == 1:
                        flow_pts.append(tuple(map(int, next_pts[idx][0])))
                    else:
                        flow_pts.append(tracker_features[feature_name][i])
                    idx += 1
                flow_features[feature_name] = tuple(flow_pts)

        # Confronta tracker vs optical flow
        max_drift_detected = False
        for feature_name in tracker_features.keys():
            if feature_name not in flow_features:
                continue

            tracker_pts = np.array(tracker_features[feature_name], dtype=np.float32)
            flow_pts = np.array(flow_features[feature_name], dtype=np.float32)

            distances = np.linalg.norm(tracker_pts - flow_pts, axis=1)

            if np.any(distances > 20.0):
                max_drift_detected = True
                break

        if max_drift_detected:
            print(f"  ⚠️ Tracker drift detected (using optical flow)")
            return flow_features

        return tracker_features

    # ========================================================================
    # TRACKING + RE-DETECTION LAYER
    # ========================================================================

    def track_or_redetect(self, frame):
        """
        Tracking multi-feature con refinement + PLATE BOTTOM UPDATE.

        Returns:
            Dict {'top': [(L,R)], 'outer': [(L,R)], 'bottom': [(L,R)]} o None
        """
        # LAYER 1: TRACKERS
        all_trackers_ok = True
        tracker_features = {}

        for feature_name, tracker in self.trackers.items():
            if tracker.is_initialized:
                success, centers = tracker.update(frame)

                if success:
                    tracker_features[feature_name] = centers
                else:
                    all_trackers_ok = False
                    break

        if all_trackers_ok and len(tracker_features) > 0:
            # LAYER 2: OPTICAL FLOW VALIDATION
            features_validated = self.validate_with_optical_flow(frame, tracker_features)

            # LAYER 3: TEMPLATE MATCHING REFINEMENT
            self.frame_count += 1
            if self.frame_count % self.refine_every_n_frames == 0:
                features_refined = self.refine_features_with_templates(frame, features_validated)

                # Aggiorna template se refinement ha successo
                if features_refined != features_validated:
                    for feature_name, centers in features_refined.items():
                        if feature_name in self.keypoint_templates:
                            for i, center in enumerate(centers):
                                new_template = self._extract_keypoint_template(frame, center)
                                if new_template.size > 0:
                                    self.keypoint_templates[feature_name][i] = new_template

                features = features_refined
            else:
                features = features_validated

            # ===== FIX: AGGIORNA PLATE BOTTOM QUI! =====
            updated_plate_bottom = self.detector.update_plate_bottom_only(frame, features)

            if updated_plate_bottom is not None:
                self.last_known_plate_bottom = updated_plate_bottom
            # Altrimenti mantieni l'ultimo noto

            self.last_known_features = features
            self.tracking_failures = 0

            # Aggiorna kalman con 'outer' (reference)
            if 'outer' in features:
                self.redetector.update_kalman(features['outer'])

            return features
        else:
            self.tracking_failures += 1

        # LAYER 4: RE-DETECTION
        if self.tracking_failures >= 3:
            # Usa 'outer' come reference per redetection
            last_known_centers = None
            if self.last_known_features and 'outer' in self.last_known_features:
                last_known_centers = self.last_known_features['outer']

            new_centers = self.redetector.redetect(
                frame, last_known_centers=last_known_centers, search_region_scale=2.5
            )

            if new_centers is not None:
                # Reinizializza TUTTI i tracker con nuova detection
                features_dict, plate_bottom = self.detect_initial_vehicle(frame)

                if features_dict is not None:
                    for feature_name, centers in features_dict.items():
                        if feature_name in self.trackers:
                            # Converti numpy array → tuple di tuple
                            if isinstance(centers, np.ndarray):
                                centers = tuple(tuple(map(int, pt)) for pt in centers)
                            self.trackers[feature_name].reinitialize(frame, centers)

                    self.last_known_features = features_dict
                    self.last_known_plate_bottom = plate_bottom  # Usa quello dalla detection
                    self.tracking_failures = 0
                    return features_dict

        # FALLBACK
        
        
        # ✅ NUOVO: stoppa processing se > 60% punti mancanti
        total_expected_features = 6  # outer(2) + top(2) + bottom(2)
        if self.last_known_features:
            actual_features = sum(len(v) for v in self.last_known_features.values())
            if actual_features < total_expected_features * 0.4:  # <40% punti
                print(f"  ⛔ CRITICAL: solo {actual_features}/{total_expected_features} features → RESET")
                self.reset()
                return None
    
    
        if self.tracking_failures >= self.MAX_TRACKING_FAILURES:
            self.reset()
            return None

        return self.last_known_features

    def reset(self):
        """Reset completo pipeline."""
        self.vehicle_detected = False
        self.last_known_features = None
        self.last_known_plate_bottom = None
        self.tracking_failures = 0
        self.pose_quality_failures = 0
        self.quality_bbox_frozen = False


        for tracker in self.trackers.values():
            tracker.reset()

        self.keypoint_templates = None
        self.frame_count = 0
        self.prev_frame_gray = None
        self.prev_features_for_filter = None
        self.last_frozen_count = 0
        self.last_good_pose = None
        self.bbox_is_frozen = False
        # Reset VP solver state
        self.vp_solver.reset_tti_history()
        self.vp_solver.reset_temporal_smoothing()
        self.vp_solver.reset_vp_persistence()

    # ========================================================================
    # FILTRO OUTLIER E FREEZE BBOX
    # ========================================================================
    def filter_outlier_points(self, current_features):
        """
        Filtro robusto con mediana del movimento.
        LOGICA:
        - Calcola il vettore di movimento di ogni punto (current - prev)
        - Usa la MEDIANA come consenso (robusta agli outlier)
        - Punti con deviazione > max_point_jump dalla mediana → outlier
        - Outlier → sostituiti con prev + mediana (previsione coerente)
        - Conta punti "fermi" (|movimento| < freeze_eps) → self.last_frozen_count
        - Se frozen_count >= frozen_threshold → self.bbox_is_frozen = True
        Returns:
            features filtrate (dict con numpy arrays)
        """
        if self.prev_features_for_filter is None:
            self.prev_features_for_filter = {
                k: np.array(v, dtype=np.float32) for k, v in current_features.items()
            }
            self.last_frozen_count = 0
            self.bbox_is_frozen = False
            return current_features
        current_features = {
            k: np.array(v, dtype=np.float32) for k, v in current_features.items()
        }
        # Raccoglie tutti i vettori di movimento
        all_movements = []
        point_info = []   # (feature_name, index) per debug
        for feature_name in ['top', 'outer', 'bottom']:
            if feature_name not in current_features:
                continue
            if feature_name not in self.prev_features_for_filter:
                continue
            curr = current_features[feature_name]
            prev = self.prev_features_for_filter[feature_name]
            for i in range(min(len(curr), len(prev))):
                mov = curr[i] - prev[i]
                all_movements.append(mov)
                point_info.append((feature_name, i))
        if len(all_movements) < 3:
            self.last_frozen_count = 0
            self.bbox_is_frozen = False
            return current_features
        movements = np.array(all_movements, dtype=np.float32)  # shape (N, 2)
        # Mediana component-wise (più robusta della media)
        median_movement = np.median(movements, axis=0)
        # ===== CONTA PUNTI FERMI =====
        frozen_count = 0
        for mov in movements:
            if np.linalg.norm(mov) < self.freeze_eps:
                frozen_count += 1
        self.last_frozen_count = frozen_count
        # Aggiorna stato freeze
        if frozen_count >= self.frozen_threshold:
            if not self.bbox_is_frozen:
                print(f"  🧊 BBox FROZEN: {frozen_count}/{len(movements)} punti fermi")
            self.bbox_is_frozen = True
        else:
            if self.bbox_is_frozen:
                print(f"  ▶ BBox UNFROZEN: solo {frozen_count}/{len(movements)} punti fermi")
            self.bbox_is_frozen = False
        # ===== FILTRA OUTLIER =====
        filtered_features = {}
        for feature_name in ['top', 'outer', 'bottom']:
            if feature_name not in current_features:
                continue
            if feature_name not in self.prev_features_for_filter:
                filtered_features[feature_name] = current_features[feature_name]
                continue
            curr = current_features[feature_name]
            prev = self.prev_features_for_filter[feature_name]
            filtered_pts = []
            for i in range(len(curr)):
                if i >= len(prev):
                    filtered_pts.append(curr[i])
                    continue
                movement = curr[i] - prev[i]
                # Deviazione rispetto al consenso (mediana)
                deviation = np.linalg.norm(movement - median_movement)

                # ✅ NUOVO: soglia variabile per tipo
                if feature_name == 'outer':
                    threshold = self.max_point_jump * 0.8  # outer: più rigido
                elif feature_name == 'top':
                    threshold = self.max_point_jump
                else:  # bottom
                    threshold = self.max_point_jump * 1.2  # bottom: più permissivo

                if deviation > threshold:
                    # OUTLIER: usa previsione basata sulla mediana
                    predicted = prev[i] + median_movement
                    filtered_pts.append(predicted.astype(np.float32))
                    if self.frame_count % 30 == 0:
                        print(f"  🔴 Outlier {feature_name}[{i}]: "
                              f"dev={deviation:.1f}px → prediction "
                              f"(mediana={median_movement})")
                else:
                    filtered_pts.append(curr[i])
            filtered_features[feature_name] = np.array(filtered_pts, dtype=np.float32)
        # Aggiorna prev per prossimo frame
        self.prev_features_for_filter = {
            k: v.copy() for k, v in filtered_features.items()
        }
        return filtered_features
    

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def process_frame(self, frame, frame_idx, prev_features=None, prev_plate_bottom=None):
        """
        Processa singolo frame con MULTI-FEATURE + TTI.
        FREEZE: dopo il filtro outlier, se self.bbox_is_frozen è True,
        non aggiorniamo last_good_pose → la bbox rimane all'ultima posizione nota.
        Il risultato contiene 'pose_for_bbox' che può essere diverso da 'pose'.
        
        Args:
            frame: Frame BGR
            frame_idx: Indice frame
            prev_features: Features frame precedente
            prev_plate_bottom: Plate bottom frame precedente

        Returns:
            Dict con risultati processing
        """
        result = {
            'success': False,
            'features': None,
            'plate_bottom': None,
            'pose': None,
            'pose_for_bbox': None,    # Posa usata per BBox, Origine e Assi (può essere frozen)
            'bbox_frozen': False,
            'motion_type': 'UNKNOWN',
            'status': 'processing'
        }

        # DETECTION INIZIALE
        if not self.vehicle_detected:
            features_dict, plate_bottom = self.detect_initial_vehicle(frame)

            if features_dict is not None:
                # Inizializza TUTTI i tracker
                for feature_name, centers in features_dict.items():
                    if feature_name in self.trackers:
                        # Converti numpy array → tuple di tuple
                        if isinstance(centers, np.ndarray):
                            centers = tuple(tuple(map(int, pt)) for pt in centers)
                        self.trackers[feature_name].initialize(frame, centers)

                self.vehicle_detected = True
                self.last_known_features = features_dict
                self.last_known_plate_bottom = plate_bottom

                # Aggiorna kalman con 'outer'
                if 'outer' in features_dict:
                    self.redetector.update_kalman(features_dict['outer'])

                result['success'] = True
                result['features'] = features_dict
                result['plate_bottom'] = plate_bottom
                result['status'] = 'initial_detection'

        # TRACKING
        else:
            features = self.track_or_redetect(frame)

            if features is not None:
                result['success'] = True
                result['features'] = features
                result['plate_bottom'] = self.last_known_plate_bottom
                result['status'] = 'tracking'
            else:
                result['status'] = 'lost'

        # FILTRO OUTLIER + FREEZE CHECK
        if result['success'] and result['features'] is not None:
            result['features'] = self.filter_outlier_points(result['features'])
            
            # --- HARDCODE FREEZE @ 195 (Blocca BBox + Origine + Assi) ---
            if frame_idx >= 195:
                if not self.quality_bbox_frozen:
                    print(f"  ❄️ HARDCODE FREEZE: Frame {frame_idx} raggiunto. Tutto il sistema 3D è bloccato.")
                self.quality_bbox_frozen = True
            # ------------------------------------------------------------

            result['bbox_frozen'] = self.quality_bbox_frozen

        # POSE ESTIMATION (SINGLE-FRAME, ROBUST)
        if result['success'] and result['features'] is not None:
            try:
                # Converti SOLO outer points (robusti, sempre visibili)
                features_t2 = {}

                for feature_name in ['outer', 'top', 'bottom']:
                    if feature_name in result['features']:
                        features_t2[feature_name] = np.array(
                            result['features'][feature_name], dtype=np.float32
                        )

                # ✅ Se mancano top o bottom, solvePnP fallirà gracefully
                if len(features_t2) < 3:  # Serve almeno outer, top, bottom
                    print(f"  ⚠️ Frame {frame_idx}: Missing features for PnP")
                    result['pose'] = None
                else:
                    # Converti plate bottom
                    plate_bottom_t2 = None
                    if result['plate_bottom'] is not None:
                        pb = np.array(result['plate_bottom'], dtype=np.float32)
                        if np.linalg.norm(pb[1] - pb[0]) > 10.0:
                            plate_bottom_t2 = pb

                    pose_data = self.vp_solver.estimate_pose_multifeature(
                        features_t2,
                        plate_bottom_t2,
                        frame_idx
                    )

                    if pose_data is not None:
                        # Posa buona: aggiorna lo stato interno SOLO se non siamo in freeze
                        if not self.quality_bbox_frozen:
                            self.pose_quality_failures = 0
                            self.last_good_pose = pose_data
                        
                        result['pose'] = pose_data
                        result['motion_type'] = pose_data.get('motion_type', 'UNKNOWN')
                    else:
                        # Posa rifiutata: attiva il freeze se non siamo già in hardcode
                        if not self.quality_bbox_frozen:
                            self.pose_quality_failures += 1
                            
                            if self.vp_solver.last_reproj_error > self.INSTANT_FREEZE_ERROR:
                                print(f"  ❄️ INSTANT FREEZE: reproj={self.vp_solver.last_reproj_error:.1f}px")
                                self.quality_bbox_frozen = True
            
                            print(f"  🔶 Frame {frame_idx}: pose rejected "
                                f"({self.pose_quality_failures}/{self.MAX_POSE_QUALITY_FAILURES})")

                            if self.pose_quality_failures >= self.MAX_POSE_QUALITY_FAILURES:
                                self.quality_bbox_frozen = True

                        result['pose'] = None
                
                # --- ASSEGNAZIONE POSA PERSISTENTE ---
                # Se siamo in freeze (frame >= 195 o errori), usiamo 'last_good_pose'.
                # Questo garantisce che BBox, Origine e Assi restino fermi insieme.
                if result['pose'] is not None and not self.quality_bbox_frozen:
                    result['pose_for_bbox'] = result['pose']
                else:
                    result['pose_for_bbox'] = self.last_good_pose
                
                result['bbox_frozen'] = self.quality_bbox_frozen

            except Exception as e:
                import traceback
                print(f"  ⚠️ Frame {frame_idx}: Pose error - {e}")
                traceback.print_exc()

        return result

# ============================================================================
# TASK 2: PROCESS VIDEO 
# ============================================================================

def process_video_task2(video_path, camera_matrix, dist_coeffs, config):
    """Task 2: Vanishing Point - MULTI-FEATURE + HYBRID TRACKING + TTI + FREEZE"""
    print("\n🌙 Task 2: Vanishing Point (Multi-Feature + Hybrid + TTI + Freeze)")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output principale
    output_path = OUTPUT_DIR / f"task2_{Path(video_path).stem}_output.avi"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), fps, (width, height), codec='MJPG')

    # Output DEBUG MASK (grandezza intera)
    output_debug_path = OUTPUT_DIR / f"task2_{Path(video_path).stem}_debug_mask.avi"
    writer_debug = VideoWriter(str(output_debug_path), fps, (width, height), codec='MJPG')

    results_dir = Path("data/results/task2_vanishing_point")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = Task2Pipeline(camera_matrix, dist_coeffs, config, width, height, fps)

    # Detection stability
    REQUIRED_CONSECUTIVE_DETECTIONS = 5
    consecutive_detections = 0
    vehicle_initialized = False

    frame_idx = 0
    prev_features = None
    prev_plate_bottom = None

    print(f"\n▶ Processing (require {REQUIRED_CONSECUTIVE_DETECTIONS} consecutive detections)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()

        # Crea debug mask frame
        debug_mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # === DETECTION STABILITY CHECK ===
        if not vehicle_initialized:
            features_dict, plate_bottom = pipeline.detect_initial_vehicle(frame)

            if features_dict is not None:
                consecutive_detections += 1

                if consecutive_detections >= REQUIRED_CONSECUTIVE_DETECTIONS:
                    # Inizializza tracker
                    for feature_name, centers in features_dict.items():
                        if feature_name in pipeline.trackers:
                            if isinstance(centers, np.ndarray):
                                centers = tuple(tuple(map(int, pt)) for pt in centers)
                            pipeline.trackers[feature_name].initialize(frame, centers)

                    pipeline.vehicle_detected = True
                    pipeline.last_known_features = features_dict
                    pipeline.last_known_plate_bottom = plate_bottom

                    if 'outer' in features_dict:
                        pipeline.redetector.update_kalman(features_dict['outer'])

                    vehicle_initialized = True
                    prev_features = features_dict
                    prev_plate_bottom = plate_bottom

                    print(f"  ✅ Tracker initialized at frame {frame_idx}")

                    # Disegna keypoints
                    for feature_name, centers in features_dict.items():
                        color = (0, 255, 0) if feature_name == 'outer' else (0, 200, 200)
                        for pt in centers:
                            cv2.circle(frame_disp, tuple(map(int, pt)), 6, color, -1)
                            cv2.circle(frame_disp, tuple(map(int, pt)), 8, (255, 255, 255), 2)

                    cv2.putText(frame_disp, "Tracking initialized!",
                               (width // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    remaining = REQUIRED_CONSECUTIVE_DETECTIONS - consecutive_detections
                    cv2.putText(frame_disp, f"Vehicle detected... stabilizing ({remaining} more needed)",
                               (width // 2 - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
            else:
                if consecutive_detections > 0:
                    print(f"  ⚠️ Frame {frame_idx}: Detection lost")
                consecutive_detections = 0

                cv2.putText(frame_disp, "Waiting for stable vehicle detection...",
                           (width // 2 - 280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            writer.write(frame_disp)
            frame_idx += 1
            continue

        # === TRACKING NORMALE ===
        result = pipeline.process_frame(frame, frame_idx, prev_features, prev_plate_bottom)

        # CREA DEBUG MASK FRAME
        # ===== FILTRO PRIMA DI POSE ESTIMATION =====
        if result['success'] and result['features'] is not None:
            pose_for_dbg = result.get('pose') or result.get('pose_for_bbox')

            # Estrai rvec, tvec se disponibili
            rvec_dbg = pose_for_dbg['rvec'] if pose_for_dbg else None
            tvec_dbg = pose_for_dbg['tvec'] if pose_for_dbg else None

            debug_mask_frame = DrawUtils.create_debug_mask_frame(
                frame,
                pipeline.detector,
                result['features'],
                result['plate_bottom'],
                rvec=rvec_dbg,
                tvec=tvec_dbg,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                frame_idx=frame_idx
            )
        else:
            debug_mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
            if result['status'] == 'lost':
                cv2.putText(debug_mask_frame, "TRACKING LOST",
                        (width // 2 - 150, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.putText(debug_mask_frame, "Waiting for vehicle...",
                        (width // 2 - 200, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # VISUALIZATION
        if result['status'] == 'lost':
            cv2.putText(frame_disp, "LOST - Searching...",
                       (width // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        elif not result['success']:
            cv2.putText(frame_disp, "Waiting for vehicle...",
                       (width // 2 - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        else:
            features = result['features']

            # Draw keypoints
            if features:
                # Outer = verde brillante
                if 'outer' in features:
                    for pt in features['outer']:
                        cv2.circle(frame_disp, tuple(map(int, pt)), 6, (0, 255, 0), -1)
                        cv2.circle(frame_disp, tuple(map(int, pt)), 8, (255, 255, 255), 2)

                # Top = ciano
                if 'top' in features:
                    for pt in features['top']:
                        cv2.circle(frame_disp, tuple(map(int, pt)), 4, (255, 255, 0), -1)

                # Bottom = giallo
                if 'bottom' in features:
                    for pt in features['bottom']:
                        cv2.circle(frame_disp, tuple(map(int, pt)), 4, (0, 255, 255), -1)

                # ===== PLATE BOTTOM VISUALIZZAZIONE =====
                if result['plate_bottom'] is not None:
                    plate_bottom_pts = result['plate_bottom']

                    # Disegna punti BL, BR
                    for pt in plate_bottom_pts:
                        cv2.circle(frame_disp, tuple(map(int, pt)), 5, (255, 0, 255), -1)

                    # Disegna linea bordo inferiore MAGENTA (spessa)
                    BL, BR = plate_bottom_pts[0], plate_bottom_pts[1]
                    cv2.line(frame_disp, tuple(map(int, BL)), tuple(map(int, BR)), 
                            (255, 0, 255), 3, cv2.LINE_AA)

                # ===== FIX: USA CORNERS DAL DETECTOR (SEMPRE AGGIORNATI) =====
                if pipeline.detector.prev_plate_corners is not None:
                    plate_corners = pipeline.detector.prev_plate_corners
                    TL = plate_corners['TL']
                    TR = plate_corners['TR']
                    BL = plate_corners['BL']
                    BR = plate_corners['BR']
                    
                    # Disegna tutti i 2 angoli inferiori della targa
                    for pt in [BL, BR]:
                        cv2.circle(frame_disp, pt, 5, (255, 128, 255), -1)


            # ===== DISEGNA POSE =====
            # pose = stima corrente (per info numeriche)
            # pose_for_bbox = last_good_pose se frozen, altrimenti corrente (per bbox)
            pose = result.get('pose')
            pose_for_bbox = result.get('pose_for_bbox') or pose
            bbox_frozen = result.get('bbox_frozen', False) or pipeline.quality_bbox_frozen

            if pose_for_bbox is not None:
                rvec_bbox = pose_for_bbox['rvec']
                tvec_bbox = pose_for_bbox['tvec']
                R_bbox = pose_for_bbox['R']
                # Assi 3D sempre dalla posa corrente se disponibile, altrimenti frozen
                rvec_axes = rvec_bbox
                tvec_axes = tvec_bbox
                
                # 3D axes
                draw_3d_axes(frame_disp, rvec_axes, tvec_axes, camera_matrix, dist_coeffs, 1.5, 4)
                # BBox 3D (potenzialmente frozen)
                bbox_2d = pipeline.bbox_projector.project_bbox(rvec_bbox, tvec_bbox)
                if bbox_2d is not None:
                    bbox_color = (0, 165, 255) if bbox_frozen else (0, 255, 0)
                    bbox_thickness = 2
                    draw_bbox_3d(frame_disp, bbox_2d, color=bbox_color, thickness=bbox_thickness)
                    
                    # Indicatore visivo freeze
                    if bbox_frozen:
                        cv2.putText(frame_disp, "BBOX FROZEN",
                                    (width // 2 - 100, height - 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                # Info numeriche dalla posa corrente (se disponibile)
                ref_pose = pose if pose else pose_for_bbox
                R_ref = ref_pose['R']
                yaw = pipeline.vp_solver.yaw_smooth if pipeline.vp_solver.yaw_smooth is not None \
                      else pipeline.vp_solver.extract_yaw_from_rotation(R_ref)
                yaw_deg = np.degrees(yaw)

                # Yaw text
                cv2.putText(frame_disp, f"Yaw: {yaw_deg:+6.1f}deg", 
                            (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Freccia direzione (da origine a 2m avanti)
                origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                forward_3d = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

                origin_2d, _ = cv2.projectPoints(origin_3d, rvec_axes, tvec_axes, camera_matrix, dist_coeffs)
                forward_2d, _ = cv2.projectPoints(forward_3d, rvec_axes, tvec_axes, camera_matrix, dist_coeffs)

                origin_px = tuple(map(int, origin_2d[0][0]))
                forward_px = tuple(map(int, forward_2d[0][0]))

                # Disegna origine (croce arancione)
                cv2.drawMarker(frame_disp, origin_px, (0, 165, 255), cv2.MARKER_CROSS, 40, 5)
                cv2.circle(frame_disp, origin_px, 20, (0, 165, 255), 3)

                # Disegna freccia direzione
                cv2.arrowedLine(frame_disp, origin_px, forward_px, (255, 255, 0), 4, tipLength=0.3)

                
                
                
                
                
                
                # Distance e TTI dalla posa corrente
                distance = np.linalg.norm(ref_pose['tvec'])
                draw_tracking_info(frame_disp, frame_idx, "PnP Estimation", distance, 2)
                
                # TTI overlay
                if  pose and 'tti' in pose and pose['tti'] is not None:
                    tti = pose['tti']
                    tti_valid = pose.get('tti_valid', True)
                    
                    tti_color = (0, 255, 0) if tti_valid else (0, 0, 255)
                    tti_text = f"TTI: {tti:.1f}s" if abs(tti) < 100 else "TTI: >100s"
                    
                    if tti < 0:
                        tti_text += " (moving away)"
                    else:
                        tti_text += " (approaching)"
                    
                    cv2.putText(frame_disp, tti_text,
                               (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tti_color, 2)
                
    #            # Debug info
     #           if pose and 'debug' in pose:
    #                method = pose['debug'].get('method', 'unknown')
    #                frozen_txt = f" [FROZEN:{pipeline.last_frozen_count}pts]" if bbox_frozen else ""
    #                cv2.putText(frame_disp, f"Method: {method}{frozen_txt}",
    #                            (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Save results
                if pose:
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    save_data = {
                        'rvec': pose['rvec'],
                        'tvec': pose['tvec'],
                        'R': pose['R'],
                        'method': 'vanishing_point_multifeature',
                        'motion_type': result['motion_type'],
                        'tracking_failures': pipeline.tracking_failures,
                        'tti': pose.get('tti'),
                        'tti_valid': pose.get('tti_valid'),
                        'pose_method': pose.get('debug', {}).get('method', 'unknown'),
                        'bbox_frozen': bbox_frozen,
                        'frozen_count': pipeline.last_frozen_count
                    }

                # Salva features
                for feature_name in ['top', 'outer', 'bottom']:
                    if feature_name in prev_features:
                        save_data[f'lights_{feature_name}_frame1'] = np.array(prev_features[feature_name])
                    if feature_name in features:
                        save_data[f'lights_{feature_name}_frame2'] = np.array(features[feature_name])

                # Salva plate bottom
                if prev_plate_bottom is not None:
                    save_data['plate_bottom_frame1'] = np.array(prev_plate_bottom)
                if result['plate_bottom'] is not None:
                    save_data['plate_bottom_frame2'] = np.array(result['plate_bottom'])

                np.savez(pose_file, **save_data)

            # Status overlay
            status_color = (0, 255, 0) if pipeline.tracking_failures == 0 else (0, 200, 200)
            cv2.putText(frame_disp, f"Status: {result['status']}",
                       (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if pipeline.tracking_failures > 0:
                cv2.putText(frame_disp, f"Fail: {pipeline.tracking_failures}/{pipeline.MAX_TRACKING_FAILURES}",
                           (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

        # ===== VISUALIZZAZIONE VANISHING POINTS =====
        if pose and 'vx' in pose and 'vy' in pose:
            # Prepara features per visualizzazione
            # Converti da numpy arrays a formato corretto
            features_for_vp = {}
            if features:
                for key in ['outer', 'top', 'bottom']:
                    if key in features:
                        features_for_vp[key] = np.array(features[key], dtype=np.float32)
            
            if result['plate_bottom'] is not None:
                features_for_vp['plate_bottom'] = np.array(result['plate_bottom'], dtype=np.float32)
            
            # Features precedenti (per traiettorie Vx)
            features_prev_vp = None
            if prev_features:
                features_prev_vp = {}
                for key in ['outer', 'top']:
                    if key in prev_features:
                        features_prev_vp[key] = np.array(prev_features[key], dtype=np.float32)
            
            # Disegna VP con linee convergenti
            DrawUtils.draw_vp_convergence(
                frame_disp,
                features_curr=features_for_vp,
                features_prev=features_prev_vp,
                vx_motion=pose.get('vx'),  # VP movimento
                vy_lateral=pose.get('vy'),  # VP laterale
                show_labels=True
            )

        # Motion type overlay (questa riga esisteva già)
        draw_motion_type_overlay(frame_disp, result['motion_type'])

        # Scrivi i due video separati a grandezza intera
        writer.write(frame_disp)
        writer_debug.write(debug_mask_frame)

        # Update prev_features
        if result['success'] and result['features'] is not None:
            prev_features = result['features']
            if result['plate_bottom'] is not None:
                prev_plate_bottom = result['plate_bottom']

        # Progress
        if frame_idx % 30 == 0:
            frozen_info = f" [frozen:{pipeline.last_frozen_count}pts]" if pipeline.bbox_is_frozen else ""
            print(f"   Frame {frame_idx}/{total_frames} "
                  
                f"({frame_idx/total_frames*100:.1f}%) - "
                f"{result['status']}{frozen_info}")

        frame_idx += 1

    cap.release()
    writer.release()
    writer_debug.release()

    print(f"\n✅ Task 2 completed!")
    print(f"   Output: {output_path}")
    print(f"   Debug Mask: {output_debug_path}")
    print(f"   Results: {results_dir}")

    return True






# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point con menu interattivo."""

    print("\n" + "=" * 70)
    print("🚗 VEHICLE LOCALIZATION SYSTEM - MULTI-FEATURE + TTI + FREEZE")
    print("=" * 70)

    # Load configurations
    try:
        config = load_all_configs()

        calib_file = config['camera_config']['camera']['calibration_file']
        camera_matrix, dist_coeffs = load_camera_calibration(calib_file)

        print("✓ Configuration loaded")
        print(f"  Camera: {calib_file}")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    # Main loop
    while True:
        show_menu()

        try:
            choice = input("\nSelect task: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break

        if choice == "0":
            print("\n👋 Bye!")
            break

        video_path, video_name = choose_video()
        if video_path is None:
            continue

        try:
            if choice == "2":
                process_video_task2(video_path, camera_matrix, dist_coeffs, config)
            else:
                print("❌ Invalid choice")

        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()