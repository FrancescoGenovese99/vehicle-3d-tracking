"""
Vehicle Localization System - VERSIONE FINALE COMPLETA CON HYBRID TRACKING
Sistema completo con menu per selezione video e metodo

Task 1: Homography da targa
Task 2: Vanishing Point da luci (HYBRID TRACKING VERSION)
Task 3: PnP diretto
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

from src.pose_estimation import homography_solver

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
    draw_3d_axes
)
from calibration.load_calibration import (
    load_camera_calibration_simple as load_camera_calibration, 
    CameraParameters
)
from utils.config_loader import load_config
from utils.motion_classifier import MotionClassifier


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
    """
    Load all configuration files.
    """
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
    print(" 🚗 VEHICLE LOCALIZATION SYSTEM ")
    print("=" * 70)
    print("1 - Task 1: Homography (License Plate)")
    print("2 - Task 2: Vanishing Point (Tail Lights)")
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
# TASK 2: VANISHING POINT (HYBRID TRACKING VERSION)
# ============================================================================

class Task2Pipeline:
    """
    Pipeline HYBRID per Task 2: Vanishing Point Localization.
    
    ARCHITETTURA:
    1. Detection iniziale (AdvancedDetector con corner detection)
    2. Tracking (CSRT tracker)
    3. Template Matching Refinement (ogni N frame)
    4. Optical Flow Validation (drift detection)
    5. Re-detection (quando necessario)
    """
    
    def __init__(self, camera_matrix, dist_coeffs, config, frame_width, frame_height):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        detection_params = config.get('detection_params', {})
        
        # ===== COMPONENTI CORE =====
        # Detector (usa AdvancedDetector con corner detection)
        self.detector = AdvancedDetector(detection_params)
        
        # Tracker
        self.tracker = LightTracker(detection_params)
        
        # Re-detection (usa detector base + selector)
        basic_detector = LightDetector(detection_params)
        selector = CandidateSelector(detection_params, frame_width, frame_height)
        self.redetector = RedetectionManager(basic_detector, selector, detection_params)
        
        # Pose estimation
        self.vp_solver = VanishingPointSolver(camera_matrix, config['vehicle_model'])
        
        # 3D projection
        cam_params = CameraParameters(camera_matrix, dist_coeffs)
        self.bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
        
        # ===== HYBRID TRACKING STATE =====
        self.vehicle_detected = False
        self.last_known_centers = None
        self.tracking_failures = 0
        self.MAX_TRACKING_FAILURES = 5
        
        # Template matching
        self.keypoint_templates = None
        self.refine_every_n_frames = 3
        self.frame_count = 0
        
        # Optical flow
        self.prev_frame_gray = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        print("  ✓ Task2Pipeline initialized (HYBRID)")
        print(f"    Detector: AdvancedDetector (corner detection)")
        print(f"    Tracker: {self.tracker.tracker_type.value}")
        print(f"    Template Matching: every {self.refine_every_n_frames} frames")
        print(f"    Optical Flow: enabled")
        print(f"    Kalman: {self.redetector.enable_kalman}")
    
    # ========================================================================
    # DETECTION LAYER
    # ========================================================================
    
    def detect_initial_vehicle(self, frame):
        """
        Detection iniziale del veicolo usando AdvancedDetector.
        
        Returns:
            Numpy array (2, 2): [[left_x, left_y], [right_x, right_y]] o None
        """
        tail_lights, templates = self.detector.detect_tail_lights_with_templates(frame)
        
        if tail_lights is not None:
            # Salva template per tracking futuro
            self.keypoint_templates = templates
            return tail_lights
        
        return None
    
    # ========================================================================
    # TEMPLATE MATCHING LAYER
    # ========================================================================
    
    def _extract_keypoint_template(self, frame: np.ndarray, center: tuple, 
                                   size: int = 25) -> np.ndarray:
        """
        Estrae template attorno a un keypoint.
        
        Args:
            frame: Frame BGR
            center: Centro (x, y)
            size: Dimensione template
        
        Returns:
            Template grayscale (size x size)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cx, cy = center
        half = size // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(gray.shape[1], cx + half + 1)
        y2 = min(gray.shape[0], cy + half + 1)
        
        template = gray[y1:y2, x1:x2].copy()
        
        # Assicura dimensione fissa
        if template.shape[0] < size or template.shape[1] < size:
            template = cv2.resize(template, (size, size))
        
        return template
    
    def _refine_keypoint_with_template(self, frame: np.ndarray, approx_center: tuple,
                                       template: np.ndarray, search_radius: int = 25) -> tuple:
        """
        Raffina keypoint usando template matching locale.
        
        Args:
            frame: Frame corrente
            approx_center: Posizione approssimativa dal tracker
            template: Template del keypoint
            search_radius: Raggio ricerca (pixel)
        
        Returns:
            (x, y) raffinato o approx_center se fallisce
        """
        if template is None or template.size == 0:
            return approx_center
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cx, cy = approx_center
        
        # ROI di ricerca
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(w, cx + search_radius)
        y2 = min(h, cy + search_radius)
        
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return approx_center
        
        # Template matching
        try:
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Threshold: accetta solo match confidenti
            if max_val > 0.55:
                # Converti coordinate locali → globali
                refined_x = x1 + max_loc[0] + template.shape[1] // 2
                refined_y = y1 + max_loc[1] + template.shape[0] // 2
                
                # Sanity check: non troppo lontano dall'approssimazione
                dist = np.sqrt((refined_x - cx)**2 + (refined_y - cy)**2)
                if dist < search_radius * 1.5:
                    return (refined_x, refined_y)
        except cv2.error:
            pass
        
        return approx_center
    
    def refine_centers_with_templates(self, frame: np.ndarray, 
                                      tracker_centers: tuple) -> tuple:
        """
        Raffina centri usando template matching.
        
        Args:
            frame: Frame corrente
            tracker_centers: Centri dal tracker
        
        Returns:
            Centri raffinati
        """
        if self.keypoint_templates is None or len(self.keypoint_templates) != 2:
            return tracker_centers
        
        refined = []
        for i, (center, template) in enumerate(zip(tracker_centers, self.keypoint_templates)):
            refined_center = self._refine_keypoint_with_template(frame, center, template, search_radius=50)
            refined.append(refined_center)
        
        return tuple(refined)
    
    # ========================================================================
    # OPTICAL FLOW LAYER
    # ========================================================================
    
    def validate_with_optical_flow(self, frame: np.ndarray, tracker_centers: tuple):
        """
        Valida centri usando optical flow per drift detection.
        
        Args:
            frame: Frame corrente
            tracker_centers: Centri dal tracker
        
        Returns:
            Centri validati o None se drift eccessivo
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame_gray is None or self.last_known_centers is None:
            self.prev_frame_gray = gray
            return tracker_centers
        
        # Calcola optical flow dai centri precedenti
        prev_pts = np.array(self.last_known_centers, dtype=np.float32).reshape(-1, 1, 2)
        
        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, gray, prev_pts, None, **self.lk_params
            )
        except cv2.error:
            self.prev_frame_gray = gray
            return tracker_centers
        
        self.prev_frame_gray = gray
        
        if next_pts is None or status is None:
            return tracker_centers
        
        # Estrai punti validi
        good_new = next_pts[status == 1]
        
        if len(good_new) < 2:
            return tracker_centers
        
        # Converti a tuple
        flow_centers = tuple(tuple(map(int, pt)) for pt in good_new)
        
        # Confronta optical flow con tracker
        tracker_pts = np.array(tracker_centers, dtype=np.float32)
        flow_pts = np.array(flow_centers, dtype=np.float32)
        
        # Calcola distanze
        distances = np.linalg.norm(tracker_pts - flow_pts, axis=1)
        
        # Se drift > 20px, segnala problema
        max_drift = 20.0
        if np.any(distances > max_drift):
            print(f"  ⚠️ Tracker drift detected: {distances} px (using optical flow)")
            return flow_centers
        
        return tracker_centers
    
    # ========================================================================
    # TRACKING + RE-DETECTION LAYER
    # ========================================================================
    
    def track_or_redetect(self, frame):
        """
        Tracking con multi-layer refinement.
        
        Pipeline:
        1. Tracker update
        2. Optical flow validation (drift check)
        3. Template matching refinement (ogni N frame)
        4. Re-detection (se necessario)
        
        Returns:
            Tuple ((left_x, left_y), (right_x, right_y)) o None
        """
        # LAYER 1: TRACKER
        if self.tracker.is_initialized:
            success, centers = self.tracker.update(frame)
            
            if success:
                # LAYER 2: OPTICAL FLOW VALIDATION
                centers_validated = self.validate_with_optical_flow(frame, centers)
                
                # LAYER 3: TEMPLATE MATCHING REFINEMENT (ogni N frame)
                self.frame_count += 1
                if self.frame_count % self.refine_every_n_frames == 0:
                    centers_refined = self.refine_centers_with_templates(frame, centers_validated)
                    
                    # Aggiorna template se il refinement ha avuto successo
                    if centers_refined != centers_validated:
                        # Re-estrai template aggiornati
                        for i, center in enumerate(centers_refined):
                            new_template = self._extract_keypoint_template(frame, center)
                            if new_template.size > 0:
                                self.keypoint_templates[i] = new_template
                    
                    centers = centers_refined
                else:
                    centers = centers_validated
                
                self.last_known_centers = centers
                self.tracking_failures = 0
                self.redetector.update_kalman(centers)
                return centers
            else:
                self.tracking_failures += 1
        
        # LAYER 4: RE-DETECTION
        if self.tracker.needs_redetection() or self.tracking_failures >= 3:
            new_centers = self.redetector.redetect(
                frame, last_known_centers=self.last_known_centers, search_region_scale=2.5
            )
            
            if new_centers is not None:
                # Converti a numpy per compatibilità
                new_centers_np = np.array(new_centers, dtype=np.float32).reshape(2, 2)
                
                # Reinizializza tracker
                self.tracker.reinitialize(frame, new_centers)
                
                # Re-estrai template
                self.keypoint_templates = []
                for i in range(2):
                    center = tuple(map(int, new_centers_np[i]))
                    template = self._extract_keypoint_template(frame, center)
                    self.keypoint_templates.append(template)
                
                self.last_known_centers = new_centers
                self.tracking_failures = 0
                return new_centers
            else:
                self.tracking_failures += 1
        
        # FALLBACK: usa ultima posizione nota se non troppi failure
        if self.tracking_failures >= self.MAX_TRACKING_FAILURES:
            self.reset()
            return None
        
        return self.last_known_centers
    
    def reset(self):
        """Reset completo pipeline."""
        self.vehicle_detected = False
        self.last_known_centers = None
        self.tracking_failures = 0
        self.tracker.reset()
        self.keypoint_templates = None
        self.frame_count = 0
        self.prev_frame_gray = None
    
    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================
    
    def process_frame(self, frame, frame_idx, prev_centers=None):
        """
        Processa singolo frame.
        
        Args:
            frame: Frame BGR
            frame_idx: Indice frame
            prev_centers: Centri frame precedente (per vanishing point)
        
        Returns:
            Dict con risultati processing
        """
        result = {
            'success': False,
            'centers': None,
            'pose': None,
            'motion_type': 'UNKNOWN',
            'status': 'processing'
        }
        
        # DETECTION INIZIALE
        if not self.vehicle_detected:
            centers_np = self.detect_initial_vehicle(frame)
            
            if centers_np is not None:
                # Converti numpy (2,2) → tuple di tuple per compatibilità tracker
                centers = (tuple(map(int, centers_np[0])), tuple(map(int, centers_np[1])))
                
                self.vehicle_detected = True
                self.last_known_centers = centers
                self.tracker.initialize(frame, centers)
                self.redetector.update_kalman(centers)
                
                result['success'] = True
                result['centers'] = centers
                result['status'] = 'initial_detection'
        
        # TRACKING
        else:
            centers = self.track_or_redetect(frame)
            
            if centers is not None:
                result['success'] = True
                result['centers'] = centers
                result['status'] = 'tracking'
            else:
                result['status'] = 'lost'
        
        # POSE ESTIMATION (se abbiamo 2 frame consecutivi)
        if result['success'] and prev_centers is not None and result['centers'] is not None:
            try:
                # Converti tuple → numpy array PRIMA di chiamare estimate_pose
                prev_centers_np = np.array(prev_centers, dtype=np.float32).reshape(2, 2)
                curr_centers_np = np.array(result['centers'], dtype=np.float32).reshape(2, 2)
                
                pose_data = self.vp_solver.estimate_pose(prev_centers_np, curr_centers_np, frame_idx)
                
                if pose_data is not None:
                    result['pose'] = pose_data
                    result['motion_type'] = pose_data.get('motion_type', 'UNKNOWN')
            
            except Exception as e:
                print(f"  ⚠️ Frame {frame_idx}: Pose error - {e}")
        
        return result


def process_video_task2(video_path, camera_matrix, dist_coeffs, config):
    """Task 2: Vanishing Point - HYBRID TRACKING VERSION"""
    print("\n🌙 Task 2: Vanishing Point (Hybrid Tracking)")
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
    
    output_path = OUTPUT_DIR / f"task2_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    results_dir = Path("data/results/task2_vanishing_point")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = Task2Pipeline(camera_matrix, dist_coeffs, config, width, height)
    
    frame_idx = 0
    prev_centers = None
    
    print(f"\n▶ Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_disp = frame.copy()
        result = pipeline.process_frame(frame, frame_idx, prev_centers)
        
        # VISUALIZATION
        if result['status'] == 'lost':
            cv2.putText(frame_disp, "LOST - Searching...",
                       (width // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        elif not result['success']:
            cv2.putText(frame_disp, "Waiting for vehicle...",
                       (width // 2 - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        else:
            centers = result['centers']
            
            # Draw keypoints
            for pt in centers:
                cv2.circle(frame_disp, pt, 6, (0, 255, 0), -1)
                cv2.circle(frame_disp, pt, 8, (255, 255, 255), 2)
            
            # Draw pose se disponibile
            if result['pose'] is not None:
                pose = result['pose']
                rvec, tvec = pose['rvec'], pose['tvec']
                Vx, Vy = pose.get('Vx'), pose.get('Vy')
                
                # Vanishing points
                if prev_centers is not None:
                    prev_centers_np = np.array(prev_centers, dtype=np.float32).reshape(2, 2)
                    curr_centers_np = np.array(centers, dtype=np.float32).reshape(2, 2)
                    
                    if Vx is not None and Vy is not None:
                        draw_vanishing_points_complete(
                            frame_disp, prev_centers_np, curr_centers_np, Vx, Vy,
                            dot_product=pose.get('dot_product', 0),
                            show_lines=True, show_labels=True
                        )
                
                # 3D axes
                draw_3d_axes(frame_disp, rvec, tvec, camera_matrix, dist_coeffs, 1.5, 4)
                
                # 3D bbox
                bbox_2d = pipeline.bbox_projector.project_bbox(rvec, tvec)
                if bbox_2d is not None:
                    draw_bbox_3d(frame_disp, bbox_2d, color=(0, 255, 0), thickness=2)
                
                # Info overlay
                distance = np.linalg.norm(tvec)
                draw_tracking_info(frame_disp, frame_idx, "Vanishing Point", distance, 2)
                
                # Save pose
                pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                np.savez(pose_file,
                        rvec=rvec, tvec=tvec, R=pose['R'],
                        method='vanishing_point',
                        motion_type=result['motion_type'],
                        Vx=Vx, Vy=Vy,
                        dot_product=pose.get('dot_product'),
                        lights_frame1=np.array(prev_centers) if prev_centers else None,
                        lights_frame2=np.array(centers),
                        tracking_failures=pipeline.tracking_failures)
            
            # Status overlay
            status_color = (0, 255, 0) if pipeline.tracking_failures == 0 else (0, 200, 200)
            cv2.putText(frame_disp, f"Status: {result['status']}",
                       (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            if pipeline.tracking_failures > 0:
                cv2.putText(frame_disp, f"Fail: {pipeline.tracking_failures}/{pipeline.MAX_TRACKING_FAILURES}",
                           (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        
        # Motion type overlay
        draw_motion_type_overlay(frame_disp, result['motion_type'])
        
        # Write frame
        writer.write(frame_disp)
        
        # Update prev_centers
        if result['success'] and result['centers'] is not None:
            prev_centers = result['centers']
        
        # Progress
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) - {result['status']}")
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    print(f"\n✅ Task 2 completed!")
    print(f"   Output: {output_path}")
    
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
            plate_corners_dict = detector.detect_plate_corners(frame, tail_lights)
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
    return True


# ============================================================================
# TASK 3: PNP
# ============================================================================
def process_video_pnp(video_path, camera_matrix, dist_coeffs, config):
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
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main entry point con menu interattivo."""
    
    print("\n" + "=" * 70)
    print("🚗 VEHICLE LOCALIZATION SYSTEM")
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
                process_video_pnp(video_path, camera_matrix, dist_coeffs, config)
            else:
                print("❌ Invalid choice")
        
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()                
                
            
            
            
            
            
            
            
            
            
            
            