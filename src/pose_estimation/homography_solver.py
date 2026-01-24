"""
Homography Solver - Task 1
Stima posa da omografia usando 4 punti complanari (angoli targa).

Formula: [r1 r2 t] = λ * K⁻¹ * H
dove λ è un fattore di scala per normalizzazione.

VERSIONE CORRETTA con coordinate consistenti
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple


class HomographySolver:
    """
    Risolve il Task 1: Localizzazione da omografia (4 punti complanari).
    """

    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)

        vehicle_data = vehicle_model.get('vehicle', {})
        plate_data = vehicle_data.get('license_plate', {})
        corners = plate_data.get('corners', {})

        self.plate_corners_3d = np.array([
            corners.get('top_left', [-0.27, 0.26, 0.455]),
            corners.get('top_right', [-0.27, -0.26, 0.455]),
            corners.get('bottom_right', [-0.27, -0.26, 0.345]),
            corners.get('bottom_left', [-0.27, 0.26, 0.345])
        ], dtype=np.float32)

        self.plate_width = plate_data.get('width', 0.52)
        self.plate_height = plate_data.get('height', 0.11)

        self._validate_plate_corners()

        self.method = cv2.RANSAC
        self.ransac_threshold = 5.0

    def _validate_plate_corners(self):
        x_coords = self.plate_corners_3d[:, 0]
        if not np.allclose(x_coords, x_coords[0], atol=1e-3):
            print(f"⚠️ Warning: Targa non su piano verticale! X = {x_coords}")

        width_computed = abs(self.plate_corners_3d[1, 1] - self.plate_corners_3d[0, 1])
        height_computed = abs(self.plate_corners_3d[0, 2] - self.plate_corners_3d[2, 2])

        if not np.isclose(width_computed, self.plate_width, atol=0.01):
            print("⚠️ Warning: Larghezza targa inconsistente")

        if not np.isclose(height_computed, self.plate_height, atol=0.01):
            print("⚠️ Warning: Altezza targa inconsistente")

    def compute_homography(self, image_points: np.ndarray) -> Optional[np.ndarray]:
        if image_points.shape[0] != 4:
            return None

        object_points_2d = self.plate_corners_3d[:, 1:]

        H, mask = cv2.findHomography(
            object_points_2d,
            image_points,
            method=self.method,
            ransacReprojThreshold=self.ransac_threshold
        )

        if H is None or np.sum(mask) < 4:
            return None

        return H

    def decompose_homography(
        self,
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        H_normalized = self.K_inv @ H

        r1 = H_normalized[:, 0]
        r2 = H_normalized[:, 1]
        t = H_normalized[:, 2]

        scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
        if scale < 1e-6:
            return None, None, None

        r1 /= scale
        r2 /= scale
        t /= scale

        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])

        U, _, Vt = np.linalg.svd(R)
        R_orthogonal = U @ Vt

        if np.linalg.det(R_orthogonal) < 0:
            R_orthogonal[:, 2] *= -1

        rvec, _ = cv2.Rodrigues(R_orthogonal)
        tvec = t.reshape(3, 1)

        return rvec, tvec, R_orthogonal

    def estimate_pose(
        self,
        image_points: np.ndarray,
        frame_idx: int = 0
    ) -> Optional[Dict]:

        H = self.compute_homography(image_points)
        if H is None:
            return None

        rvec, tvec, R = self.decompose_homography(H)
        if rvec is None:
            return None

        distance = np.linalg.norm(tvec)
        if distance < 0.5 or distance > 100:
            return None

        reproj_error = self.compute_reprojection_error(image_points, rvec, tvec)

        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'H': H,
            'distance': float(distance),
            'reprojection_error': reproj_error,
            'motion_type': 'UNKNOWN',
            'method': 'homography',
            'frame': frame_idx
        }

    def compute_reprojection_error(
        self,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> float:

        projected, _ = cv2.projectPoints(
            self.plate_corners_3d,
            rvec,
            tvec,
            self.K,
            None
        )

        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected, axis=1)
        return float(np.mean(errors))

    def get_plate_corners_3d(self) -> np.ndarray:
        return self.plate_corners_3d.copy()

    def validate_plate_geometry(self, image_points: np.ndarray) -> bool:
        top_width = np.linalg.norm(image_points[1] - image_points[0])
        bottom_width = np.linalg.norm(image_points[2] - image_points[3])
        left_height = np.linalg.norm(image_points[3] - image_points[0])
        right_height = np.linalg.norm(image_points[2] - image_points[1])

        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2

        if avg_height < 1:
            return False

        aspect_ratio = avg_width / avg_height
        if aspect_ratio < 2.5 or aspect_ratio > 6.0:
            return False

        def cross(p1, p2, p3):
            return np.cross(p2 - p1, p3 - p2)

        signs = [cross(image_points[i],
                       image_points[(i + 1) % 4],
                       image_points[(i + 2) % 4]) for i in range(4)]

        return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def solve_homography(
    image_points_2d: np.ndarray,
    object_points_3d: np.ndarray,
    camera_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    object_points_plane = object_points_3d[:, 1:]
    H, _ = cv2.findHomography(object_points_plane, image_points_2d, cv2.RANSAC, 5.0)

    if H is None:
        return None, None

    K_inv = np.linalg.inv(camera_matrix)
    H_norm = K_inv @ H

    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]

    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1 /= scale
    r2 /= scale
    t /= scale

    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    return rvec, tvec
