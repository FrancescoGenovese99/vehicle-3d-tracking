"""
Homography Solver - Task 1
Stima posa da omografia usando 4 punti complanari (angoli targa).

Formula: [r1 r2 t] = λ * K⁻¹ * H
dove λ è un fattore di scala per normalizzazione.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple


class HomographySolver:
    """
    Risolve il Task 1: Localizzazione da omografia (4 punti complanari).
    
    Pipeline:
    1. Riceve 4 punti 2D immagine (angoli targa)
    2. Usa 4 punti 3D noti dal modello CAD
    3. Calcola omografia H
    4. Decompone H in [r1 r2 t]
    5. Ricostruisce r3 e matrice rotazione completa R
    """
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        """
        Initialize solver.
        
        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            vehicle_model: Dictionary with vehicle geometry
        """
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        # Extract 3D plate corners from vehicle model
        vehicle_data = vehicle_model.get('vehicle', {})
        plate_data = vehicle_data.get('license_plate', {})
        corners_3d = plate_data.get('corners', {})
        
        # Order: top-left, top-right, bottom-right, bottom-left
        self.plate_corners_3d = np.array([
            corners_3d.get('top_left', [-0.30, 0.25, 0.45]),
            corners_3d.get('top_right', [-0.30, -0.25, 0.45]),
            corners_3d.get('bottom_right', [-0.30, -0.25, 0.35]),
            corners_3d.get('bottom_left', [-0.30, 0.25, 0.35])
        ], dtype=np.float32)
        
        # Plate dimensions for validation
        self.plate_width = plate_data.get('width', 0.52)  # meters
        self.plate_height = plate_data.get('height', 0.11)
        
        # Homography method
        self.method = cv2.RANSAC
        self.ransac_threshold = 5.0
    
    def compute_homography(
        self,
        image_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute homography from 3D plane to 2D image.
        
        Args:
            image_points: 4 corners in image (4, 2) - ordered TL, TR, BR, BL
            
        Returns:
            Homography matrix H (3x3) or None if failed
        """
        if image_points.shape[0] != 4:
            return None
        
        # Project 3D points to 2D plane (X, Y) - ignore Z since coplanar
        # Plate is on plane X=-0.30 (constant), so we use (Y, Z) as 2D coords
        object_points_2d = self.plate_corners_3d[:, 1:]  # (4, 2) [Y, Z]
        
        # Compute homography: object_plane → image_plane
        H, mask = cv2.findHomography(
            object_points_2d,
            image_points,
            method=self.method,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        if H is None:
            return None
        
        # Verify homography quality
        inliers = np.sum(mask)
        if inliers < 4:  # Need all 4 points
            return None
        
        return H
    
    def decompose_homography(
        self,
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose homography to extract rotation and translation.
        
        Formula from task specs:
        [r1 r2 t] = K⁻¹ * H
        
        Then normalize and compute r3 = r1 × r2
        
        Args:
            H: Homography matrix (3x3)
            
        Returns:
            Tuple (rvec, tvec, R)
        """
        # Apply K inverse
        H_normalized = self.K_inv @ H
        
        # Extract columns: [r1 r2 t]
        r1 = H_normalized[:, 0]
        r2 = H_normalized[:, 1]
        t = H_normalized[:, 2]
        
        # Normalize by scale factor (mean of ||r1|| and ||r2||)
        scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
        
        if scale < 1e-6:
            # Invalid homography
            return None, None, None
        
        r1 = r1 / scale
        r2 = r2 / scale
        t = t / scale
        
        # Compute r3 as cross product (perpendicular to r1 and r2)
        r3 = np.cross(r1, r2)
        
        # Build rotation matrix
        R = np.column_stack([r1, r2, r3])
        
        # Ensure R is a valid rotation matrix (orthogonal)
        # Use SVD to get closest orthogonal matrix
        U, _, Vt = np.linalg.svd(R)
        R_orthogonal = U @ Vt
        
        # Check for reflection (det should be +1, not -1)
        if np.linalg.det(R_orthogonal) < 0:
            # Flip sign of third column
            R_orthogonal[:, 2] *= -1
        
        # Convert to Rodrigues vector
        rvec, _ = cv2.Rodrigues(R_orthogonal)
        
        # Translation vector
        tvec = t.reshape(3, 1)
        
        return rvec, tvec, R_orthogonal
    
    def estimate_pose(
        self,
        image_points: np.ndarray,
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """
        Full pose estimation pipeline for Task 1.
        
        Args:
            image_points: 4 plate corners in image (4, 2)
            frame_idx: Frame index (for logging)
            
        Returns:
            Dictionary with pose data or None
            {
                'rvec': rotation vector (3x1),
                'tvec': translation vector (3x1),
                'R': rotation matrix (3x3),
                'H': homography matrix (3x3),
                'motion_type': 'UNKNOWN' (single frame),
                'method': 'homography'
            }
        """
        # Step 1: Compute homography
        H = self.compute_homography(image_points)
        
        if H is None:
            return None
        
        # Step 2: Decompose to get pose
        rvec, tvec, R = self.decompose_homography(H)
        
        if rvec is None:
            return None
        
        # Validate pose (translation should be positive distance)
        distance = np.linalg.norm(tvec)
        if distance < 0.5 or distance > 100:  # Reasonable range: 0.5m to 100m
            return None
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'H': H,
            'distance': float(distance),
            'motion_type': 'UNKNOWN',  # Cannot determine from single frame
            'method': 'homography',
            'frame': frame_idx
        }
    
    def compute_reprojection_error(
        self,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> float:
        """
        Compute reprojection error to validate pose.
        
        Args:
            image_points: Observed 2D points (4, 2)
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            Mean reprojection error in pixels
        """
        # Project 3D plate corners
        projected, _ = cv2.projectPoints(
            self.plate_corners_3d,
            rvec,
            tvec,
            self.K,
            None  # Assuming no distortion for homography
        )
        
        projected = projected.reshape(-1, 2)
        
        # Compute error
        errors = np.linalg.norm(image_points - projected, axis=1)
        mean_error = np.mean(errors)
        
        return float(mean_error)
    
    def get_plate_corners_3d(self) -> np.ndarray:
        """
        Get 3D plate corners in vehicle frame.
        
        Returns:
            Array (4, 3) with 3D coordinates
        """
        return self.plate_corners_3d.copy()
    
    def validate_plate_geometry(
        self,
        image_points: np.ndarray
    ) -> bool:
        """
        Validate that detected corners form a valid plate shape.
        
        Args:
            image_points: 4 corners (4, 2)
            
        Returns:
            True if geometry is valid
        """
        # Check aspect ratio
        top_width = np.linalg.norm(image_points[1] - image_points[0])
        bottom_width = np.linalg.norm(image_points[2] - image_points[3])
        left_height = np.linalg.norm(image_points[3] - image_points[0])
        right_height = np.linalg.norm(image_points[2] - image_points[1])
        
        avg_width = (top_width + bottom_width) / 2
        avg_height = (left_height + right_height) / 2
        
        if avg_height < 1:  # Avoid division by zero
            return False
        
        aspect_ratio = avg_width / avg_height
        
        # Italian plate aspect ratio ~4.7:1, allow range 2.5-6.0
        if aspect_ratio < 2.5 or aspect_ratio > 6.0:
            return False
        
        # Check if corners form a convex quadrilateral
        # (all cross products should have same sign)
        def cross_product_sign(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p2
            return np.cross(v1, v2)
        
        signs = [
            cross_product_sign(image_points[i], 
                             image_points[(i+1)%4], 
                             image_points[(i+2)%4])
            for i in range(4)
        ]
        
        # All should have same sign for convex
        if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            return False
        
        return True


def solve_homography(image_points_2d: np.ndarray, 
                    object_points_3d: np.ndarray, 
                    camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone function for compatibility with existing code.
    
    Args:
        image_points_2d: 4 points in image (4, 2)
        object_points_3d: 4 points in 3D (4, 3)
        camera_matrix: K matrix (3x3)
        
    Returns:
        Tuple (rvec, tvec)
    """
    # Project 3D to plane (use Y, Z only)
    object_points_plane = object_points_3d[:, 1:]
    
    # Compute H
    H, _ = cv2.findHomography(object_points_plane, image_points_2d, cv2.RANSAC, 5.0)
    
    if H is None:
        return None, None
    
    # Decompose
    K_inv = np.linalg.inv(camera_matrix)
    H_norm = K_inv @ H
    
    r1 = H_norm[:, 0]
    r2 = H_norm[:, 1]
    t = H_norm[:, 2]
    
    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1 = r1 / scale
    r2 = r2 / scale
    t = t / scale
    
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    return rvec, tvec