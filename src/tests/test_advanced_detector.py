"""
Test del nuovo detector avanzato.
"""

import cv2
import numpy as np
from src.detection.advanced_detector import AdvancedDetector

# Carica video
VIDEO_PATH = 'data/videos/input/video1.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

# Salta a metà video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Errore caricamento frame")
    exit(1)

# Crea detector
detector = AdvancedDetector()

# Rileva tutto
keypoints = detector.detect_all(frame)

if keypoints is None:
    print("❌ Nessun keypoint rilevato")
    exit(1)

print("✓ Rilevamento completato!")
print(f"  Fari: {keypoints.tail_lights}")
print(f"  Targa angoli: {keypoints.plate_corners}")
print(f"  Targa centro: {keypoints.plate_center}")
print(f"  Confidenza: {keypoints.confidence:.2f}")

# Visualizza
overlay = frame.copy()

# Disegna fari
for i, (x, y) in enumerate(keypoints.tail_lights):
    cv2.circle(overlay, (x, y), 10, (0, 255, 255), -1)
    cv2.putText(overlay, f'L' if i == 0 else 'R', (x + 15, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Disegna targa
if keypoints.plate_corners:
    pts = np.array([
        keypoints.plate_corners['TL'],
        keypoints.plate_corners['TR'],
        keypoints.plate_corners['BR'],
        keypoints.plate_corners['BL']
    ], dtype=np.int32)
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
    
    # Angoli
    for name, (x, y) in keypoints.plate_corners.items():
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(overlay, name, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Centro targa
cv2.circle(overlay, keypoints.plate_center, 8, (255, 0, 255), -1)

# Salva
cv2.imwrite('data/test_advanced_detection.jpg', overlay)
print("\n✓ Immagine salvata: data/test_advanced_detection.jpg")
print("  Aprila da Windows per vedere il risultato!")