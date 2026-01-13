import sys
from pathlib import Path
sys.path.insert(0, '/app/src')

from scripts.vehicle_localization_system import process_video_task1
from utils.config_loader import load_config
from calibration.load_calibration import load_camera_calibration_simple as load_camera_calibration

# Video diurno con targa visibile
video_path = "data/videos/input/2.mp4"  # ← Scegli video diurno

print("🚗 Vehicle 3D Localization System - Task 1")
print("=" * 60)
print(f"📹 Video: {video_path}")

if not Path(video_path).exists():
    print(f"❌ Video non trovato")
    sys.exit(1)

# Config
config = {}
config['camera_config'] = load_config('config/camera_config.yaml')
config['detection_params'] = load_config('config/detection_params.yaml')
config['vehicle_model'] = load_config('config/vehicle_model.yaml')
config['tracking'] = config['detection_params'].get('tracking', {})

calib_file = config['camera_config']['camera']['calibration_file']
camera_matrix, dist_coeffs = load_camera_calibration(calib_file)

print("✅ Config caricate")

# PROCESSA
success = process_video_task1(video_path, camera_matrix, dist_coeffs, config)

if success:
    print("\n🎉 Task 1 completato!")
else:
    print("\n❌ Errore Task 1")
