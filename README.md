Vehicle 3D Localization System
Frame-by-frame estimation of the 3D pose (position + orientation) of a moving vehicle observed by a single calibrated fixed camera. The system detects and tracks rear tail lights under nighttime conditions, reconstructs the pose via PnP, and projects a 3D bounding box onto the video.
Reference vehicle: Toyota Aygo X.

Method
The system implements the nighttime localization task described in the project specifications. Tail lights are detected through HSV filtering, and three sub-keypoints are extracted per side (outer, top, bottom). Tracking is handled by CSRT trackers, validated frame-by-frame via Lucas-Kanade optical flow and periodically refined with template matching.
The 3D pose is estimated using cv2.solvePnP (SQPNP) with up to 6 2D–3D correspondences, where the 3D coordinates of each keypoint are taken from the vehicle CAD model. Translation is smoothed with EMA, rotation with SLERP over quaternions.
Vanishing points (motion Vx, lateral Vy) are computed and rendered for diagnostic purposes but do not contribute to pose estimation: on real footage they prove too unstable to be reliable.

Structure
vehicle-3d-tracking/
├── config/
│   ├── camera_config.yaml
│   ├── detection_params.yaml
│   └── vehicle_model.yaml
├── data/
│   ├── calibration/
│   │   ├── images/          # Chessboard calibration images
│   │   └── camera1.npz      # Calibrated K matrix
│   ├── videos/
│   │   ├── input/           # Input videos
│   │   └── output/          # Annotated output videos
│   └── results/
│       └── vanishing_point/ # Per-frame pose data (.npz)
├── src/
│   ├── calibration/
│   ├── detection/
│   ├── tracking/
│   ├── pose_estimation/
│   ├── visualization/
│   └── utils/
├── scripts/
│   └── vehicle_localization_system.py
├── notebooks/
├── task/
├── Dockerfile
└── docker-compose.yml

Setup with Docker
bash# Build
docker-compose build

# Start main container
docker-compose up -d vehicle-tracker
docker-compose exec vehicle-tracker bash

# Start Jupyter (optional, port 8888)
docker-compose up -d jupyter
Place the .mp4 videos to process in data/videos/input/ before running.

Usage
bashpython scripts/vehicle_localization_system.py
The program presents an interactive text menu:
1 - Recalibrate camera
2 - Process video
0 - Exit
Option 1 — recalibrates the camera from the chessboard images in data/calibration/images/. Prompts to confirm pattern size and square dimensions; overwrites the calibration file.
Option 2 — lists available videos in data/videos/input/ and runs the full pipeline on the selected one.

Output
For each processed video:

data/videos/output/{name}_output.avi — annotated video with keypoints, 3D bounding box wireframe, reference frame axes, yaw, distance, TTI, and motion type classification
data/videos/output/{name}_debug_mask.avi — diagnostic video with HSV mask and reprojection of estimated points
data/results/vanishing_point/frame_XXXX.npz — per-frame pose data: rvec, tvec, R, motion_type, tti, and 2D features


Configuration
Main parameters are in config/:

vehicle_model.yaml — vehicle dimensions and 3D keypoint coordinates in the vehicle reference frame (origin: rear axle center at ground level)
detection_params.yaml — HSV thresholds, blob parameters, tracker type (CSRT/KCF/MOSSE), bounding box padding
camera_config.yaml — calibration file path, resolution, chessboard pattern parameters


Dependencies
opencv-contrib-python==4.8.1.78
numpy==1.24.3
scipy==1.11.3
PyYAML==6.0.1
Full installation: pip install -r requirements.txt

Author
Francesco Genovese — @FrancescoGenovese99