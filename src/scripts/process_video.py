#!/usr/bin/env python3
"""
Script principale per processare un singolo video.

Pipeline completa:
1. Carica configurazioni e calibrazione camera
2. Rileva fari nel primo frame
3. Traccia fari frame-by-frame
4. Stima posa 3D con PnP
5. Proietta bounding box 3D
6. Salva video annotato e risultati

Usage:
    python scripts/process_video.py --input data/videos/input/video1.mp4 \
                                     --output data/videos/output/video1_tracked.mp4 \
                                     --save-results
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Aggiungi la directory root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs
from src.utils.data_io import save_tracked_points, save_all_poses, ensure_output_dirs
from src.calibration.load_calibration import load_camera_from_config
from src.detection.light_detector import LightDetector
from src.detection.candidate_selector import CandidateSelector
from src.tracking.tracker import LightTracker
from src.tracking.redetection import RedetectionManager
from src.pose_estimation.pnp_solver import PnPSolver
from src.pose_estimation.bbox_3d_projector import BBox3DProjector
from src.visualization.draw_utils import DrawUtils
from src.visualization.video_writer import VideoWriterManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Processa video per tracking 3D veicolo',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path del video di input'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path del video di output (default: auto-generato in data/videos/output/)'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory contenente i file di configurazione'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Salva risultati numerici (punti, pose, bbox)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Mostra video in tempo reale durante processing'
    )
    
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        default=True,
        help='Disegna bounding box 3D sul video'
    )
    
    parser.add_argument(
        '--draw-axes',
        action='store_true',
        help='Disegna assi del sistema di riferimento'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Numero massimo di frame da processare (per debug)'
    )
    
    return parser.parse_args()


def process_video(args):
    """
    Processa il video completo.
    
    Args:
        args: Argomenti da command line
        
    Returns:
        0 se successo, 1 se errore
    """
    # ========== SETUP ==========
    print("="*60)
    print("VEHICLE 3D TRACKING - PROCESSING")
    print("="*60)
    
    # Carica configurazioni
    print("\n[1/9] Caricamento configurazioni...")
    configs = load_all_configs(args.config_dir)
    detection_config = configs['detection_params']
    vehicle_config = configs['vehicle_model']
    camera_config = configs['camera_config']
    
    # Carica calibrazione camera
    print("[2/9] Caricamento calibrazione camera...")
    camera_params = load_camera_from_config(camera_config)
    print(f"  Camera: fx={camera_params.fx:.1f}, fy={camera_params.fy:.1f}")
    
    # Apri video
    print("[3/9] Apertura video...")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌ Impossibile aprire video: {args.input}")
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  FPS: {fps}, Frames: {frame_count}, Size: {frame_width}x{frame_height}")
    
    if args.max_frames:
        frame_count = min(frame_count, args.max_frames)
        print(f"  Limitato a {frame_count} frames per debug")
    
    # Setup output
    print("[4/9] Setup output...")
    if args.output is None:
        video_name = Path(args.input).stem
        args.output = f"data/videos/output/{video_name}_tracked.mp4"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.save_results:
        ensure_output_dirs()
    
    # Inizializza moduli
    print("[5/9] Inizializzazione moduli...")
    detector = LightDetector(detection_config)
    selector = CandidateSelector(detection_config, frame_width)
    tracker = LightTracker(detection_config)
    redetection_mgr = RedetectionManager(detector, selector, detection_config)
    pnp_solver = PnPSolver(camera_params, vehicle_config, 
                           camera_config.get('pnp', {}))
    bbox_projector = BBox3DProjector(camera_params, vehicle_config)
    
    # ========== PROCESSING ==========
    print("[6/9] Detection iniziale...")
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Impossibile leggere primo frame")
        return 1
    
    # Rileva fari nel primo frame
    # Cerca il primo frame con fari validi
    print("[6/9] Ricerca frame con fari visibili...")
    initial_centers = None
    initial_frame_idx = 0
    max_search_frames = 150  # Cerca nei primi 5 secondi (150 frames a 30fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Torna all'inizio

    for search_idx in range(max_search_frames):
        ret, search_frame = cap.read()
        if not ret:
            break
        
        candidates, mask = detector.detect_tail_lights(search_frame)
        
        if len(candidates) >= 2:  # Almeno 2 candidati
            centers = selector.get_tail_light_centers(candidates)
            
            if centers is not None:
                # Verifica che i fari siano "ragionevoli"
                dx = abs(centers[1][0] - centers[0][0])
                dy = abs(centers[1][1] - centers[0][1])
                
                # Filtri aggiuntivi:
                # - Distanza orizzontale ragionevole (100-500px tipicamente)
                # - Allineamento verticale buono (dy < 50px)
                # - Entrambi nella metà inferiore del frame (fari in basso)
                if (100 < dx < 500 and 
                    dy < 50 and 
                    centers[0][1] > frame_height * 0.3 and
                    centers[1][1] > frame_height * 0.3):
                    
                    initial_centers = centers
                    initial_frame_idx = search_idx
                    first_frame = search_frame.copy()
                    print(f"  ✓ Fari rilevati al frame {initial_frame_idx}: {initial_centers}")
                    print(f"    Distanza: {dx:.0f}px, Allineamento: {dy:.0f}px")
                    break
        
        if search_idx % 30 == 0:
            print(f"  Ricerca in corso... frame {search_idx}/{max_search_frames}")

    if initial_centers is None:
        print("❌ Nessun faro valido trovato nei primi 5 secondi")
        print("   Suggerimenti:")
        print("   - Verifica che i fari siano visibili nel video")
        print("   - Usa i notebook per ottimizzare i parametri HSV")
        print("   - Controlla che il veicolo sia presente nel video")
        return 1
    
    print(f"  ✓ Fari rilevati: {initial_centers}")
    
    # Inizializza tracker
    tracker.initialize(first_frame, initial_centers)
    
    # Setup video writer
    print("[7/9] Inizializzazione video writer...")
    video_writer = VideoWriterManager(args.output, fps, (frame_width, frame_height))
    
    # Storage per risultati
    all_tracked_points = []
    all_poses = []
    trajectory = []
    
    print("[8/9] Processing frames...")
    frame_idx = 0
    
    # Processa primo frame
    image_points = np.array(list(initial_centers), dtype=np.float32)
    success, rvec, tvec = pnp_solver.solve(image_points)
    
    if success:
        all_poses.append((rvec, tvec))
        all_tracked_points.append(image_points)
        
        # Calcola centro per traiettoria
        center = tuple(image_points.mean(axis=0).astype(int))
        trajectory.append(center)
    
    # Annota primo frame
    annotated = first_frame.copy()
    annotated = DrawUtils.draw_tracked_points(annotated, initial_centers)
    annotated = DrawUtils.draw_tracking_status(annotated, "TRACKING", frame_idx)
    
    if success and args.draw_bbox:
        projected_bbox = bbox_projector.project_bbox(rvec, tvec)
        annotated = bbox_projector.draw_bbox_on_frame(annotated, projected_bbox)
        
        if args.draw_axes:
            annotated = bbox_projector.draw_axes(annotated, rvec, tvec)
    
    video_writer.write_frame(annotated)
    
    # Progress bar
    pbar = tqdm(total=frame_count-1, desc="Processing", unit="frame")
    
    # Loop principale
    frames_lost = 0
    
    for frame_idx in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update tracking
        tracking_success, current_centers = tracker.update(frame)
        
        if tracking_success:
            status = "TRACKING"
            frames_lost = 0
            
            # Stima posa
            image_points = np.array(list(current_centers), dtype=np.float32)
            pnp_success, rvec, tvec = pnp_solver.solve(image_points)
            
            if pnp_success:
                all_poses.append((rvec, tvec))
                all_tracked_points.append(image_points)
                
                # Update traiettoria
                center = tuple(image_points.mean(axis=0).astype(int))
                trajectory.append(center)
                
                # Update Kalman in redetection manager
                redetection_mgr.update_kalman(current_centers)
        else:
            frames_lost += 1
            
            # Controlla se serve re-detection
            if tracker.needs_redetection():
                status = "REDETECTING"
                
                # Esegui re-detection
                new_centers = redetection_mgr.redetect(
                    frame, 
                    tracker.get_current_centers()
                )
                
                if new_centers:
                    # Re-inizializza tracker
                    tracker.reinitialize(frame, new_centers)
                    current_centers = new_centers
                    tracking_success = True
                    status = "TRACKING"
                    frames_lost = 0
                else:
                    status = "LOST"
                    current_centers = tracker.get_current_centers()
            else:
                status = "LOST"
                current_centers = tracker.get_current_centers()
        
        # Annota frame
        annotated = frame.copy()
        
        if current_centers:
            annotated = DrawUtils.draw_tracked_points(annotated, current_centers)
            
            # Se abbiamo una posa valida, disegna bbox
            if len(all_poses) > 0 and args.draw_bbox:
                rvec, tvec = all_poses[-1]
                projected_bbox = bbox_projector.project_bbox(rvec, tvec)
                annotated = bbox_projector.draw_bbox_on_frame(annotated, projected_bbox)
                
                if args.draw_axes:
                    annotated = bbox_projector.draw_axes(annotated, rvec, tvec)
        
        # Disegna traiettoria
        if len(trajectory) > 1:
            annotated = DrawUtils.draw_trajectory(annotated, trajectory)
        
        annotated = DrawUtils.draw_tracking_status(annotated, status, frame_idx)
        
        # Scrivi frame
        video_writer.write_frame(annotated)
        
        # Visualizza se richiesto
        if args.visualize:
            cv2.imshow('Vehicle Tracking', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  Interrotto dall'utente")
                break
        
        pbar.update(1)
    
    pbar.close()
    
    # ========== CLEANUP ==========
    print("[9/9] Finalizzazione...")
    cap.release()
    video_writer.release()
    
    if args.visualize:
        cv2.destroyAllWindows()
    
    # Salva risultati
    if args.save_results and len(all_tracked_points) > 0:
        video_name = Path(args.input).stem
        
        # Salva punti tracciati
        tracked_array = np.array(all_tracked_points)
        save_tracked_points(
            tracked_array,
            f"data/results/tracked_points/{video_name}_points.npz",
            metadata={'video': args.input, 'fps': fps}
        )
        
        # Salva pose
        if len(all_poses) > 0:
            save_all_poses(
                all_poses,
                "data/results/poses",
                video_name
            )
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("PROCESSING COMPLETATO!")
    print("="*60)
    print(f"Frames processati: {frame_idx + 1}/{frame_count}")
    print(f"Pose stimate: {len(all_poses)}")
    print(f"Video output: {args.output}")
    
    if args.save_results:
        print(f"Risultati salvati in: data/results/")
    
    print("="*60)
    
    return 0


def main():
    """Main function."""
    args = parse_args()
    
    try:
        return process_video(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrotto dall'utente")
        return 1
    except Exception as e:
        print(f"\n\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())