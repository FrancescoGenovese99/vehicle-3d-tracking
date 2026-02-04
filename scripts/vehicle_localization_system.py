"""
Vehicle Localization System - VERSIONE FINALE MULTI-FEATURE + TTI
Sistema completo con menu per selezione video e metodo

Task 1: Homography da targa
Task 2: Vanishing Point da luci (MULTI-FEATURE + HYBRID TRACKING + TTI)
Task 3: PnP diretto
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
    print("1 - Task 1: Homography (License Plate)")
    print("2 - Task 2: Vanishing Point (Multi-Feature + Hybrid Tracking + TTI)")
    print("3 - Task 3: PnP Direct")
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
# TASK 2: VANISHING POINT (MULTI-FEATURE + HYBRID TRACKING + TTI)
# ============================================================================

class Task2Pipeline:
    """
    Pipeline MULTI-FEATURE HYBRID per Task 2.
    
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
        self.vp_solver = VanishingPointSolver(camera_matrix, config['vehicle_model'])
        
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
        
        print("  ✓ Task2Pipeline initialized (MULTI-FEATURE HYBRID + TTI)")
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
    
    def _extract_keypoint_template(self, frame: np.ndarray, center: tuple, 
                                   size: int = 25) -> np.ndarray:
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
    
    def _refine_keypoint_with_template(self, frame: np.ndarray, approx_center: tuple,
                                       template: np.ndarray, search_radius: int = 25) -> tuple:
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
    
    def refine_features_with_templates(self, frame: np.ndarray, 
                                       tracker_features: dict) -> dict:
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
            for i, (center, template) in enumerate(zip(centers, templates)):
                refined_center = self._refine_keypoint_with_template(frame, center, template, search_radius=50)
                refined.append(refined_center)
            
            refined_features[feature_name] = tuple(refined)
        
        return refined_features
    
    # ========================================================================
    # OPTICAL FLOW LAYER
    # ========================================================================
    
    def validate_with_optical_flow(self, frame: np.ndarray, tracker_features: dict):
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
        
        for tracker in self.trackers.values():
            tracker.reset()
        
        self.keypoint_templates = None
        self.frame_count = 0
        self.prev_frame_gray = None
        
        # Reset VP solver state
        self.vp_solver.reset_tti_history()
        self.vp_solver.reset_temporal_smoothing()
        self.vp_solver.reset_vp_persistence()

    
    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================
    
    def process_frame(self, frame, frame_idx, prev_features=None, prev_plate_bottom=None):
        """
        Processa singolo frame con MULTI-FEATURE + TTI.
        
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
        
       # POSE ESTIMATION (SINGLE-FRAME, ROBUST)
        if result['success'] and result['features'] is not None:

            try:
                # Converti SOLO outer points (robusti, sempre visibili)
                features_t2 = {}

                if 'outer' in result['features']:
                    features_t2['outer'] = np.array(
                        result['features']['outer'], dtype=np.float32
                    )

                
                # Converti plate bottom (SOLO frame corrente)
                plate_bottom_t2 = None
                if result['plate_bottom'] is not None:
                    pb = np.array(result['plate_bottom'], dtype=np.float32)
                    if np.linalg.norm(pb[1] - pb[0]) > 10.0:  # evita degenerazioni
                        plate_bottom_t2 = pb


                pose_data = self.vp_solver.estimate_pose_multifeature(
                    features_t2,
                    plate_bottom_t2,
                    frame_idx
                )

              
                if pose_data is not None:
                    result['pose'] = pose_data
                    result['motion_type'] = pose_data.get('motion_type', 'UNKNOWN')
            
            except Exception as e:
                print(f"  ⚠️ Frame {frame_idx}: Pose error - {e}")
        
        return result


"""
ESTRATTO main_pipeline.py - Solo le modifiche necessarie per Task 2

MODIFICHE:
1. Nessuna modifica a Task2Pipeline (già corretta)
2. Aggiornamento visualizzazione in process_video_task2()
3. Fix disegno plate bottom
"""

# ============================================================================
# TASK 2: VANISHING POINT - ESTRATTO MODIFICHE
# ============================================================================

def process_video_task2(video_path, camera_matrix, dist_coeffs, config):
    """Task 2: Vanishing Point - MULTI-FEATURE + HYBRID TRACKING + TTI"""
    print("\n🌙 Task 2: Vanishing Point (Multi-Feature + Hybrid + TTI)")
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
    
    # Output DEBUG MASK
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
        if result['success'] and result['features'] is not None:
            debug_mask_frame = DrawUtils.create_debug_mask_frame(
                frame,
                pipeline.detector,
                result['features'],
                result['plate_bottom']
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
                
                # ===== FIX: PLATE BOTTOM VISUALIZZAZIONE =====
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
                    
                    # Disegna tutti i 4 angoli
                    for pt in [TL, TR, BL, BR]:
                        cv2.circle(frame_disp, pt, 5, (255, 128, 255), -1)
                                
            # Draw pose se disponibile
            if result['pose'] is not None:
                pose = result['pose']
                rvec, tvec = pose['rvec'], pose['tvec']
                
                
                # 3D axes
                draw_3d_axes(frame_disp, rvec, tvec, camera_matrix, dist_coeffs, 1.5, 4)
                
                # 3D bbox
                bbox_2d = pipeline.bbox_projector.project_bbox(rvec, tvec)
                if bbox_2d is not None:
                    draw_bbox_3d(frame_disp, bbox_2d, color=(0, 255, 0), thickness=2)
                
                # Info overlay
                distance = np.linalg.norm(tvec)
                draw_tracking_info(frame_disp, frame_idx, "Vanishing Point (Multi-Feature)", distance, 2)
                
                # TTI overlay
                if 'tti' in pose and pose['tti'] is not None:
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
                
                # Multi-feature debug info
                if 'debug' in pose:
                    debug = pose['debug']
                    method = debug.get('method', 'unknown')
                    cv2.putText(frame_disp, f"Pose method: {method}",
                                (10, height - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                  
                
                # Save pose
                pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                save_data = {
                    'rvec': rvec,
                    'tvec': tvec,
                    'R': pose['R'],
                    'method': 'vanishing_point_multifeature',
                    'motion_type': result['motion_type'],
                    'tracking_failures': pipeline.tracking_failures,
                    'tti': pose.get('tti'),
                    'tti_valid': pose.get('tti_valid'),
                    'pose_method': pose.get('debug', {}).get('method', 'unknown')
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
        
        # Motion type overlay
        draw_motion_type_overlay(frame_disp, result['motion_type'])
        
        # Write frames
        writer.write(frame_disp)
        writer_debug.write(debug_mask_frame)
        
        # Update prev_features
        if result['success'] and result['features'] is not None:
            prev_features = result['features']
            if result['plate_bottom'] is not None:
                prev_plate_bottom = result['plate_bottom']
        
        # Progress
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) - {result['status']}")
        
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
# TASK 1: HOMOGRAPHY
# ============================================================================

def process_video_task1(video_path, camera_matrix, dist_coeffs, config):
    """Task 1: Homography da targa."""
    print("\n🔷 Task 1: Homography (License Plate)")
    print("=" * 70)
    
    detector = AdvancedDetector(config.get('detection_params', {}))
    homography_solver = HomographySolver(camera_matrix, config['vehicle_model'])
    cam_params = CameraParameters(camera_matrix, dist_coeffs)
    bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
    motion_classifier = MotionClassifier()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   {width}x{height} @ {fps}fps, {total_frames} frames")
    
    output_path = OUTPUT_DIR / f"task1_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    results_dir = Path("data/results/task1_homography")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    prev_rvec = None
    vehicle_detected = False
    
    print(f"\n▶ Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        motion_type = "UNKNOWN"
        
        tail_lights = detector.detect_tail_lights(frame)
        plate_corners = None
        
        if tail_lights is not None:
            plate_corners_dict = detector.detect_plate_corners(frame, {'outer': tail_lights})
            if plate_corners_dict:
                plate_corners = np.array([
                    plate_corners_dict['TL'], plate_corners_dict['TR'],
                    plate_corners_dict['BR'], plate_corners_dict['BL']
                ], dtype=np.float32)
                
                if not vehicle_detected:
                    vehicle_detected = True
                    print(f"  ✓ License plate detected at frame {frame_idx}")
        
        if not vehicle_detected:
            cv2.putText(frame_display, "Waiting for license plate...", 
                       (width // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            writer.write(frame_display)
            frame_idx += 1
            continue
        
        if plate_corners is not None:
            try:
                pose_data = homography_solver.estimate_pose(plate_corners, frame_idx)
                if pose_data:
                    rvec, tvec = pose_data['rvec'], pose_data['tvec']
                    
                    if prev_rvec is not None:
                        motion_type, _ = motion_classifier.classify_from_pose_change(rvec, prev_rvec)
                    prev_rvec = rvec.copy()
                    
                    bbox_2d = bbox_projector.project_bbox(rvec, tvec)
                    if bbox_2d is not None:
                        draw_bbox_3d(frame_display, bbox_2d, color=(255, 165, 0), thickness=2)
                    
                    for pt in plate_corners:
                        cv2.circle(frame_display, tuple(pt.astype(int)), 8, (0, 255, 255), -1)
                    cv2.polylines(frame_display, [plate_corners.astype(int)], True, (0, 255, 255), 2)
                    
                    distance = np.linalg.norm(tvec)
                    draw_tracking_info(frame_display, frame_idx, "Homography", distance, 4)
                    
                    # Save results
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(pose_file,
                            rvec=rvec, tvec=tvec, R=pose_data['R'],
                            method='homography',
                            motion_type=motion_type,
                            plate_corners=plate_corners)
            
            except Exception as e:
                print(f"  ⚠️ Frame {frame_idx}: {e}")
        
        draw_motion_type_overlay(frame_display, motion_type)
        writer.write(frame_display)
        
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"\n✅ Task 1 completed!")
    print(f"   Output: {output_path}")
    return True


# ============================================================================
# TASK 3: PNP
# ============================================================================

def process_video_task3(video_path, camera_matrix, dist_coeffs, config):
    """Task 3: PnP diretto."""
    print("\n🔧 Task 3: PnP Direct")
    print("=" * 70)
    
    detector = AdvancedDetector(config.get('detection_params', {}))
    cam_params = CameraParameters(camera_matrix, dist_coeffs)
    pnp_solver = PnPSolver(cam_params, config['vehicle_model'],
                          config['camera_config'].get('methods', {}).get('pnp', {}))
    bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
    motion_classifier = MotionClassifier()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   {width}x{height} @ {fps}fps, {total_frames} frames")
    
    output_path = OUTPUT_DIR / f"task3_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    results_dir = Path("data/results/task3_pnp")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    prev_rvec = None
    vehicle_detected = False
    
    print(f"\n▶ Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        tail_lights = detector.detect_tail_lights(frame)
        lights = tail_lights if tail_lights is not None else None
        
        if not vehicle_detected and lights is not None:
            vehicle_detected = True
            print(f"  ✓ Vehicle detected at frame {frame_idx}")
        
        if not vehicle_detected:
            cv2.putText(frame_display, "Waiting for vehicle...", 
                       (width // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            writer.write(frame_display)
            frame_idx += 1
            continue
        
        motion_type = "UNKNOWN"
        
        if lights is not None:
            try:
                success, rvec, tvec = pnp_solver.solve(lights)
                if success:
                    if prev_rvec is not None:
                        motion_type, _ = motion_classifier.classify_from_pose_change(rvec, prev_rvec)
                    prev_rvec = rvec.copy()
                    
                    bbox_2d = bbox_projector.project_bbox(rvec, tvec)
                    if bbox_2d is not None:
                        draw_bbox_3d(frame_display, bbox_2d, color=(255, 0, 255), thickness=2)
                    
                    for light in lights:
                        cv2.circle(frame_display, tuple(map(int, light)), 5, (0, 255, 0), -1)
                    
                    distance = np.linalg.norm(tvec)
                    draw_tracking_info(frame_display, frame_idx, "PnP", distance, 2)
                    
                    # Save results
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(pose_file,
                            rvec=rvec, tvec=tvec,
                            method='pnp',
                            motion_type=motion_type,
                            tail_lights=lights)
            
            except Exception as e:
                print(f"  ⚠️ Frame {frame_idx}: {e}")
        
        draw_motion_type_overlay(frame_display, motion_type)
        writer.write(frame_display)
        
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"\n✅ Task 3 completed!")
    print(f"   Output: {output_path}")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point con menu interattivo."""
    
    print("\n" + "=" * 70)
    print("🚗 VEHICLE LOCALIZATION SYSTEM - MULTI-FEATURE + TTI")
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
            if choice == "1":
                process_video_task1(video_path, camera_matrix, dist_coeffs, config)
            elif choice == "2":
                process_video_task2(video_path, camera_matrix, dist_coeffs, config)
            elif choice == "3":
                process_video_task3(video_path, camera_matrix, dist_coeffs, config)
            else:
                print("❌ Invalid choice")
        
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()