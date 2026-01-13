"""
Vehicle 3D Localization System - Main Entry Point
VERSIONE 2 - Con visualizzazione completa (bbox sempre, motion overlay, vanishing point lines)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# from ui.interactive_menu import select_video_and_method
from utils.config_loader import load_config
from calibration.load_calibration import load_camera_calibration_simple as load_camera_calibration
from visualization.video_writer import VideoWriter
import cv2
import numpy as np


def process_video_task1(video_path: str, camera_matrix, dist_coeffs, config):
    """Process video using Task 1: Homography method."""
    print("\n🔷 Task 1: Localizzazione da Omografia (4 punti targa)")
    print("=" * 60)
    
    from detection.plate_detector import PlateDetector
    from pose_estimation.homography_solver import HomographySolver
    from pose_estimation.bbox_3d_projector import BBox3DProjector
    from visualization.draw_utils import (
        draw_tracking_info,
        draw_motion_type_overlay,
        draw_bbox_3d
    )
    from utils.motion_classifier import MotionClassifier
    
    # Initialize
    plate_detector = PlateDetector(config['detection_params'])
    homography_solver = HomographySolver(camera_matrix, config['vehicle_model'])
    from calibration.load_calibration import CameraParameters
    cam_params = CameraParameters(camera_matrix, dist_coeffs)
    bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
    motion_classifier = MotionClassifier()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Errore: impossibile aprire video {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"   Risoluzione: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame totali: {total_frames}")
    
    output_path = Path("data/videos/output") / f"task1_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    results_dir = Path("data/results/task1_homography")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n▶ Inizio processing...")
    print(f"💾 Output: {output_path}")
    
    frame_idx = 0
    prev_rvec = None
    detection_failures = 0
    max_failures = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        motion_type = "UNKNOWN"
        
        # Detect plate corners
        plate_corners = plate_detector.detect_plate_corners(frame)
        
        if plate_corners is not None:
            detection_failures = 0  # Reset failure counter
            
            try:
                # Estimate pose from homography
                pose_data = homography_solver.estimate_pose(plate_corners, frame_idx)
                
                if pose_data is not None:
                    rvec = pose_data['rvec']
                    tvec = pose_data['tvec']
                    R = pose_data['R']
                    H = pose_data['H']
                    
                    # Classify motion
                    if prev_rvec is not None:
                        motion_type, _ = motion_classifier.classify_from_pose_change(rvec, prev_rvec)
                    else:
                        motion_type = "TRANSLATION"
                    
                    prev_rvec = rvec.copy()
                    
                    # ✅ Draw bbox
                    bbox_2d = bbox_projector.project_bbox(rvec, tvec)
                    if bbox_2d is not None:
                        draw_bbox_3d(frame_display, bbox_2d, color=(255, 165, 0), thickness=2)  # Orange
                    
                    # Draw plate corners
                    for i, corner in enumerate(plate_corners):
                        pt = tuple(corner.astype(int))
                        cv2.circle(frame_display, pt, 8, (0, 255, 255), -1)  # Yellow circles
                        
                        labels = ['TL', 'TR', 'BR', 'BL']
                        cv2.putText(frame_display, labels[i], (pt[0] + 10, pt[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Draw plate boundary
                    cv2.polylines(frame_display, [plate_corners.astype(int)], True, (0, 255, 255), 2)
                    
                    # Info
                    distance = np.linalg.norm(tvec)
                    draw_tracking_info(frame_display, frame_idx, "Homography", distance, 4)
                    
                    # Reprojection error
                    reproj_error = homography_solver.compute_reprojection_error(
                        plate_corners, rvec, tvec
                    )
                    cv2.putText(frame_display, f"Reproj Error: {reproj_error:.2f}px",
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(
                        pose_file,
                        rvec=rvec,
                        tvec=tvec,
                        R=R,
                        H=H,
                        method='homography',
                        motion_type=motion_type,
                        plate_corners=plate_corners,
                        reprojection_error=reproj_error
                    )
            
            except Exception as e:
                print(f"⚠️ Frame {frame_idx}: Errore homography - {e}")
                detection_failures += 1
        else:
            # Plate not detected
            detection_failures += 1
            
            if detection_failures > max_failures:
                cv2.putText(frame_display, "⚠️ PLATE DETECTION LOST",
                           (width // 2 - 200, height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # ✅ Motion overlay
        draw_motion_type_overlay(frame_display, motion_type)
        
        writer.write(frame_display)
        
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"   Frame {frame_idx}/{total_frames} ({progress:.1f}%)")
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    print(f"\n✅ Processing completato!")
    print(f"   Frame processati: {frame_idx}")
    print(f"   Output salvato: {output_path}")
    
    return True


def process_video_task2(video_path: str, camera_matrix, dist_coeffs, config):
    """Process video using Task 2: Vanishing point method."""    # Helper: convert format for tracker
    def to_tracker_format(lights):
        """Convert numpy array to tuple format for tracker."""
        if lights is None:
            return None
        return tuple(map(tuple, lights))
    
    def to_numpy_format(lights):
        """Convert tracker output to numpy array."""
        if lights is None:
            return None
        if isinstance(lights, tuple):
            return np.array(lights)
        return lights
    

    print("\n🌙 Task 2: Localizzazione da Punto di Fuga (luci notturne)")
    print("=" * 60)
    
    # Import Task 2 modules
    from detection.light_detector import LightDetector
    from tracking.tracker import VehicleTracker
    from pose_estimation.vanishing_point_solver import VanishingPointSolver
    from pose_estimation.bbox_3d_projector import BBox3DProjector
    from visualization.draw_utils import (
        draw_tracking_info,
        draw_motion_type_overlay,
        draw_vanishing_point_lines,
        draw_bbox_3d
    )
    
    # Initialize components
    detector = LightDetector(config['detection_params'])
    tracker = VehicleTracker(config['tracking'])
    vp_solver = VanishingPointSolver(camera_matrix, config['vehicle_model'])
    from calibration.load_calibration import CameraParameters
    cam_params = CameraParameters(camera_matrix, dist_coeffs)
    bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Errore: impossibile aprire video {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"   Risoluzione: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame totali: {total_frames}")
    
    # Output video writer
    output_path = Path("data/videos/output") / f"task2_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    # Results directory
    results_dir = Path("data/results/task2_vanishing_point")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n▶ Inizio processing...")
    print(f"💾 Output: {output_path}")
    
    frame_idx = 0
    prev_lights = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        
        # Detect or track lights
        if frame_idx == 0:
            lights = detector.detect_tail_lights(frame)
            if lights is not None:
                # Converti da numpy array a tuple di tuple
                tracker.initialize(frame, to_tracker_format(lights))
                prev_lights = lights
        else:
            lights = tracker.update(frame)
            lights = to_numpy_format(lights)
            
            if lights is None:
                lights = detector.detect_tail_lights(frame)
                if lights is not None:
                    tracker.initialize(frame, to_tracker_format(lights))
        
        # Default motion type
        motion_type = "UNKNOWN"
        vanishing_point = None
        
        # If we have lights in current and previous frame, calculate pose
        if lights is not None and prev_lights is not None:
            try:
                # Calculate vanishing point and pose
                pose_data = vp_solver.estimate_pose(
                    prev_lights,
                    lights,
                    frame_idx
                )
                
                if pose_data is not None:
                    rvec = pose_data['rvec']
                    tvec = pose_data['tvec']
                    R = pose_data['R']
                    vanishing_point = pose_data.get('vanishing_point')
                    motion_type = pose_data.get('motion_type', 'UNKNOWN')
                    
                    # ✅ SEMPRE proietta bounding box 3D
                    bbox_2d = bbox_projector.project_bbox(rvec, tvec)
                    
                    if bbox_2d is not None:
                        draw_bbox_3d(frame_display, bbox_2d, color=(0, 255, 0), thickness=2)
                    
                    # Draw vanishing point lines and trajectories
                    draw_vanishing_point_lines(
                        frame_display,
                        prev_lights,
                        lights,
                        vanishing_point
                    )
                    
                    # Draw info bottom-left
                    distance = np.linalg.norm(tvec)
                    draw_tracking_info(
                        frame_display,
                        frame_idx,
                        "Vanishing Point",
                        distance,
                        len(lights)
                    )
                    
                    # Save pose data
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(
                        pose_file,
                        rvec=rvec,
                        tvec=tvec,
                        R=R,
                        method='vanishing_point',
                        motion_type=motion_type,
                        vanishing_point=vanishing_point if vanishing_point is not None else np.array([])
                    )
            
            except Exception as e:
                print(f"⚠️ Frame {frame_idx}: Errore calcolo posa - {e}")
        
        elif lights is not None:
            # Solo luci correnti, nessuna posa
            # Disegna solo le luci
            for light in lights:
                cv2.circle(frame_display, tuple(light.astype(int)), 6, (0, 255, 0), -1)
        
        # ✅ SEMPRE mostra motion type overlay (in alto al centro, GIALLO)
        draw_motion_type_overlay(frame_display, motion_type)
        
        # Update previous lights
        if lights is not None:
            prev_lights = lights.copy()
        
        # Write frame
        writer.write(frame_display)
        
        # Progress
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"   Frame {frame_idx}/{total_frames} ({progress:.1f}%)")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    writer.release()
    
    print(f"\n✅ Processing completato!")
    print(f"   Frame processati: {frame_idx}")
    print(f"   Output salvato: {output_path}")
    print(f"   Dati posa: {results_dir}")
    
    return True


def process_video_pnp(video_path: str, camera_matrix, dist_coeffs, config):
    """Process video using PnP method (direct comparison)."""
    print("\n🔧 Metodo PnP: Risoluzione Diretta (confronto)")
    print("=" * 60)
    
    from detection.light_detector import LightDetector
    from tracking.tracker import VehicleTracker
    from pose_estimation.pnp_full_solver import PnPSolver
    from pose_estimation.bbox_3d_projector import BBox3DProjector
    from visualization.draw_utils import (
        draw_tracking_info,
        draw_motion_type_overlay,
        draw_bbox_3d
    )
    from utils.motion_classifier import MotionClassifier
    
    # Initialize
    detector = LightDetector(config['detection_params'])
    tracker = VehicleTracker(config['tracking'])
    
    # PnP solver initialization (needs different signature)
    from calibration.load_calibration import CameraParameters
    cam_params = CameraParameters(camera_matrix, dist_coeffs)
    pnp_solver = PnPSolver(cam_params, config['vehicle_model'], 
                          config['camera_config'].get('methods', {}).get('pnp', {}))
    
    bbox_projector = BBox3DProjector(cam_params, config['vehicle_model'])
    motion_classifier = MotionClassifier()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Errore: impossibile aprire video {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"   Risoluzione: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame totali: {total_frames}")
    
    output_path = Path("data/videos/output") / f"pnp_{Path(video_path).stem}_output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = VideoWriter(str(output_path), fps, (width, height))
    
    results_dir = Path("data/results/pnp_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n▶ Inizio processing...")
    print(f"💾 Output: {output_path}")
    
    frame_idx = 0
    prev_rvec = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        
        if frame_idx == 0:
            lights = detector.detect_tail_lights(frame)
            if lights is not None:
                tracker.initialize(frame, to_tracker_format(lights))
        else:
            lights = tracker.update(frame)
            lights = to_numpy_format(lights)
            if lights is None:
                lights = detector.detect_tail_lights(frame)
                if lights is not None:
                    tracker.initialize(frame, to_tracker_format(lights))
        
        motion_type = "UNKNOWN"
        
        if lights is not None:
            try:
                success, rvec, tvec = pnp_solver.solve(lights)
                
                if success and rvec is not None and tvec is not None:
                    # Classify motion
                    if prev_rvec is not None:
                        motion_type, _ = motion_classifier.classify_from_pose_change(rvec, prev_rvec)
                    else:
                        motion_type = "TRANSLATION"
                    
                    prev_rvec = rvec.copy()
                    
                    # ✅ SEMPRE bbox
                    bbox_2d = bbox_projector.project_bbox(rvec, tvec)
                    if bbox_2d is not None:
                        draw_bbox_3d(frame_display, bbox_2d, color=(255, 0, 255), thickness=2)
                    
                    # Draw lights
                    for light in lights:
                        cv2.circle(frame_display, tuple(light.astype(int)), 5, (0, 255, 0), -1)
                    
                    distance = np.linalg.norm(tvec)
                    draw_tracking_info(frame_display, frame_idx, "PnP Direct", distance, len(lights))
                    
                    R, _ = cv2.Rodrigues(rvec)
                    pose_file = results_dir / f"frame_{frame_idx:04d}.npz"
                    np.savez(pose_file, rvec=rvec, tvec=tvec, R=R, method='pnp', motion_type=motion_type)
            
            except Exception as e:
                print(f"⚠️ Frame {frame_idx}: Errore PnP - {e}")
        
        # ✅ SEMPRE motion overlay
        draw_motion_type_overlay(frame_display, motion_type)
        
        writer.write(frame_display)
        
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"   Frame {frame_idx}/{total_frames} ({progress:.1f}%)")
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    print(f"\n✅ Processing completato!")
    print(f"   Frame processati: {frame_idx}")
    print(f"   Output salvato: {output_path}")
    
    return True


def main():
    """Main entry point with interactive menu."""
    print("🚗 Vehicle 3D Localization System")
    print("=" * 60)
    
    video_path, method = select_video_and_method()
    
    if not video_path or not method:
        print("\n✖ Operazione annullata dall'utente.")
        return
    
    print(f"\n✓ Video: {video_path}")
    print(f"✓ Metodo: {method}")
    
    print("\n📂 Caricamento configurazioni...")
    config = {}
    config['camera_config'] = load_config('config/camera_config.yaml')
    config['detection_params'] = load_config('config/detection_params.yaml')
    config['vehicle_model'] = load_config('config/vehicle_model.yaml')
    config['tracking'] = config['detection_params'].get('tracking', {})
    
    calib_file = config['camera_config']['camera']['calibration_file']
    camera_matrix, dist_coeffs = load_camera_calibration(calib_file)
    
    print(f"✓ Matrice camera caricata: {camera_matrix.shape}")
    
    if method == "homography":
        success = process_video_task1(video_path, camera_matrix, dist_coeffs, config)
    elif method == "vanishing_point":
        success = process_video_task2(video_path, camera_matrix, dist_coeffs, config)
    elif method == "pnp":
        success = process_video_pnp(video_path, camera_matrix, dist_coeffs, config)
    else:
        print(f"❌ Metodo sconosciuto: {method}")
        return
    
    if success:
        print("\n🎉 Sistema completato con successo!")
    else:
        print("\n❌ Errore durante il processing.")


if __name__ == "__main__":
    main()