"""
BBox 3D Projector

Projects the 3D bounding box of the tracked vehicle onto the image plane.
Uses the 'outer' tail light points as pose reference, consistent with VanishingPointSolver.

Coordinate system (vehicle frame):
  Origin : center of the rear axle at ground level
  X      : forward (positive)
  Y      : left    (positive)
  Z      : up      (positive)
"""

import numpy as np
import cv2
from typing import Optional


class BBox3DProjector:
    """
    Projects a vehicle 3D bounding box onto a 2D image given a pose estimate.

    The 8 box vertices are defined in the vehicle coordinate frame with the
    origin at the rear axle center (ground level).  Once a pose (rvec, tvec)
    is available from VanishingPointSolver, call project_bbox() to obtain the
    corresponding 2D pixel coordinates.
    """

    def __init__(self, camera_params, vehicle_model: dict):
        """
        Args:
            camera_params : object exposing .camera_matrix and .dist_coeffs
            vehicle_model : dict loaded from vehicle_model.yaml
        """
        self.camera_matrix = camera_params.camera_matrix
        self.dist_coeffs   = camera_params.dist_coeffs

        vehicle_data = vehicle_model.get('vehicle', {})

        # --- Tail light reference points (outer) ---
        # 'outer' is used as the pose reference point by VanishingPointSolver,
        # so we store the same coordinates here for consistency.
        tail_lights     = vehicle_data.get('tail_lights', {})
        left_outer      = tail_lights.get('left',  {}).get('outer', [-0.30,  0.71, 1.04])
        right_outer     = tail_lights.get('right', {}).get('outer', [-0.30, -0.71, 1.04])

        self.lights_x = left_outer[0]   # distance behind rear axle (m)
        self.lights_y = left_outer[1]   # lateral offset (m)
        self.lights_z = left_outer[2]   # height above ground (m)

        # --- Vehicle dimensions ---
        dimensions    = vehicle_data.get('dimensions', {})
        self.length   = dimensions.get('length', 3.70)
        self.width    = dimensions.get('width',  1.74)
        self.height   = dimensions.get('height', 1.525)

        # --- Bounding box vertices in vehicle frame ---
        # Bottom face (Z = 0, ground level)
        bottom_rear_right  = np.array([-0.54,  -0.87, 0.0],   dtype=np.float32)
        bottom_rear_left   = np.array([-0.54,   0.87, 0.0],   dtype=np.float32)
        bottom_front_left  = np.array([ 3.16,   0.87, 0.0],   dtype=np.float32)
        bottom_front_right = np.array([ 3.16,  -0.87, 0.0],   dtype=np.float32)

        # Top face (Z = vehicle height)
        top_rear_right     = np.array([-0.54,  -0.87, 1.525], dtype=np.float32)
        top_rear_left      = np.array([-0.54,   0.87, 1.525], dtype=np.float32)
        top_front_left     = np.array([ 3.16,   0.87, 1.525], dtype=np.float32)
        top_front_right    = np.array([ 3.16,  -0.87, 1.525], dtype=np.float32)

        # Vertex ordering matches the edge-drawing convention in draw_utils.py:
        # indices 0-3 = bottom face, 4-7 = top face (same lateral order)
        self.bbox_3d = np.array([
            bottom_rear_right,   # 0
            bottom_rear_left,    # 1
            bottom_front_left,   # 2
            bottom_front_right,  # 3
            top_rear_right,      # 4
            top_rear_left,       # 5
            top_front_left,      # 6
            top_front_right,     # 7
        ], dtype=np.float32)

        print(f"[BBox3DProjector] Initialized")
        print(f"  Vehicle dimensions : {self.length:.2f} x {self.width:.2f} x {self.height:.2f} m")
        print(f"  Outer light ref    : [{self.lights_x:.2f}, +/-{self.lights_y:.2f}, {self.lights_z:.2f}] m")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project_bbox(self, rvec: np.ndarray, tvec: np.ndarray) -> Optional[np.ndarray]:
        """
        Project the 8 bounding box vertices onto the image.

        Args:
            rvec : rotation vector    (3, 1)  from solvePnP
            tvec : translation vector (3, 1)  from solvePnP

        Returns:
            np.ndarray of shape (8, 2) with pixel coordinates,
            or None if projection fails.
        """
        try:
            points_2d, _ = cv2.projectPoints(
                self.bbox_3d,
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            return points_2d.reshape(-1, 2)

        except Exception as e:
            print(f"[BBox3DProjector] ⚠️ Projection failed: {e}")
            return None

    def get_bbox_vertices_3d(self) -> np.ndarray:
        """Return a copy of the 8 bounding box vertices in the vehicle frame."""
        return self.bbox_3d.copy()