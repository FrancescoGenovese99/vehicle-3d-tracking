"""
Processing video con detector avanzato (6 punti).
"""

import cv2
import numpy as np
from tqdm import tqdm
from src.detection.advanced_detector import AdvancedDetector
from src.pose_estimation.pnp_solver import PnPSolver
from src.pose_estimation.bbox_3d_projector import BBox3DProjector
from src.calibration.load_calibration import load_camera_from_config
from src.utils.config_loader import load_all_configs
from src.visualization.video_writer import VideoWriterManager

# Config
configs = load_all_configs('config')
camera_params = load_camera_from_config(configs['camera_config'])

# Detector
detector = AdvancedDetector()

# PnP
pnp_solver = PnPSolver(camera_params, configs['vehicle_model'], configs['camera_config'].get('pnp', {}))
bbox_projector = BBox3DProjector(camera_params, configs['vehicle_model'])

# Video
VIDEO_PATH = 'data/videos/input/video1.mp4'
OUTPUT_PATH = 'data/videos/output/video1_advanced.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = VideoWriterManager(OUTPUT_PATH, fps, (w, h))

print(f"Processing {frame_count} frames...")

for frame_idx in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    keypoints = detector.detect_all(frame)
    
    if keypoints and keypoints.confidence > 0.8:
        # Costruisci array punti 2D
        image_points = np.array([
            keypoints.tail_lights[0],
            keypoints.tail_lights[1]
        ], dtype=np.float32)
        
        # Aggiungi angoli targa se disponibili
        if keypoints.plate_corners:
            image_points = np.vstack([
                image_points,
                [keypoints.plate_corners['TL']],
                [keypoints.plate_corners['TR']],
                [keypoints.plate_corners['BL']],
                [keypoints.plate_corners['BR']]
            ])
        
        # PnP
        success, rvec, tvec = pnp_solver.solve(image_points)
        
        if success:
            # Disegna bbox 3D
            projected_bbox = bbox_projector.project_bbox(rvec, tvec)
            frame = bbox_projector.draw_bbox_on_frame(frame, projected_bbox)
            
            # Disegna keypoints
            for x, y in keypoints.tail_lights:
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            
            if keypoints.plate_corners:
                for x, y in keypoints.plate_corners.values():
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    writer.write_frame(frame)

cap.release()
writer.release()

print(f"\n✓ Video salvato: {OUTPUT_PATH}")