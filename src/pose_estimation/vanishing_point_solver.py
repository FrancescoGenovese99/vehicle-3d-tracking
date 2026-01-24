"""
Vanishing Point Solver - TYPE-SAFE VERSION
Geometria proiettiva corretta per Task 2 con gestione robusta dei tipi
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Union, List


class VanishingPointSolver:
    """
    Risolve posa usando vanishing points - VERSIONE TYPE-SAFE.
    
    GEOMETRIA CORRETTA:
    - Vx → direzione segmento luci (ASSE Y veicolo: laterale)
    - Vy → direzione movimento (ASSE X veicolo: avanti)
    - Z veicolo → perpendicolare, verso alto
    
    GESTIONE TIPI:
    - Accetta sia tuple che numpy array in input
    - Converte automaticamente a numpy internamente
    """
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        vehicle_data = vehicle_model.get('vehicle', {})
        tail_lights = vehicle_data.get('tail_lights', {})
        
        self.lights_distance_real = tail_lights.get('distance_between', 1.40)
        
        left_light = tail_lights.get('left', [-0.27, 0.70, 0.50])
        right_light = tail_lights.get('right', [-0.27, -0.70, 0.50])
        
        self.lights_height = left_light[2]
        self.lights_x_offset = left_light[0]
        
        self.tail_lights_3d = np.array([left_light, right_light], dtype=np.float32)
        
        self.perpendicularity_threshold = 0.45
        
        print(f"[VP Solver] Distance={self.lights_distance_real}m, Height={self.lights_height}m")
        print(f"[VP Solver] Perpendicularity threshold = {self.perpendicularity_threshold}")
    
    def _ensure_numpy_array(
        self,
        data: Union[Tuple, List, np.ndarray],
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Converte input a numpy array con shape specificato."""
        if isinstance(data, (tuple, list)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Tipo non supportato: {type(data)}")
        
        if arr.shape != shape:
            arr = arr.reshape(shape)
        
        return arr
    
    def calculate_vanishing_points(
        self,
        lights_frame1: Union[Tuple, np.ndarray],
        lights_frame2: Union[Tuple, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Calcola Vx e Vy."""
        lights1 = self._ensure_numpy_array(lights_frame1, (2, 2))
        lights2 = self._ensure_numpy_array(lights_frame2, (2, 2))
        
        L1, R1 = lights1[0], lights1[1]
        L2, R2 = lights2[0], lights2[1]
        
        Vx = self._line_intersection(L1, R1, L2, R2)
        Vy = self._line_intersection(L1, L2, R1, R2)
        
        if Vx is not None and not self._is_point_valid(Vx):
            Vx = None
        if Vy is not None and not self._is_point_valid(Vy):
            Vy = None
        
        return Vx, Vy
    
    def _line_intersection(self, p1, p2, p3, p4) -> Optional[np.ndarray]:
        """Intersezione rette."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return np.array([x, y], dtype=np.float32)
    
    def _is_point_valid(self, point: np.ndarray, max_dist: float = 10000.0) -> bool:
        """Valida punto."""
        if not np.isfinite(point).all():
            return False
        return np.linalg.norm(point) < max_dist
    
    def check_perpendicularity(
        self,
        Vx: np.ndarray,
        Vy: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Check K⁻¹Vx ⊥ K⁻¹Vy."""
        if threshold is None:
            threshold = self.perpendicularity_threshold
        
        dir_Vx = self.K_inv @ np.append(Vx, 1.0)
        dir_Vy = self.K_inv @ np.append(Vy, 1.0)
        
        dir_Vx /= np.linalg.norm(dir_Vx)
        dir_Vy /= np.linalg.norm(dir_Vy)
        
        dot = abs(np.dot(dir_Vx, dir_Vy))
        return dot < threshold, dot
    
    def compute_plane_pi(self, Vx, Vy) -> np.ndarray:
        """Calcola normale piano π."""
        Vx_homog = np.append(Vx, 1.0)
        Vy_homog = np.append(Vy, 1.0)
        
        l = np.cross(Vx_homog, Vy_homog)
        n_pi = self.K.T @ l
        
        return n_pi / np.linalg.norm(n_pi)
    
    def localize_light_segment_3d(
        self,
        lights_frame2: Union[Tuple, np.ndarray],
        Vx: np.ndarray,
        n_pi: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Localizza centro luci 3D."""
        lights2 = self._ensure_numpy_array(lights_frame2, (2, 2))
        
        center_2d = np.mean(lights2, axis=0)
        center_norm = self.K_inv @ np.append(center_2d, 1.0)
        
        pixel_dist = np.linalg.norm(lights2[1] - lights2[0])
        if pixel_dist < 1.0:
            return center_norm * 10.0, 10.0
        
        f_avg = (self.fx + self.fy) / 2.0
        Z = (self.lights_distance_real * f_avg) / pixel_dist
        
        center_3d = center_norm * Z
        return center_3d, Z
    
    def reconstruct_pose_from_plane(
        self,
        lights_frame2: Union[Tuple, np.ndarray],
        Vx: np.ndarray,
        Vy: np.ndarray,
        center_3d: np.ndarray,
        distance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ricostruzione posa CORRETTA.
        
        - Vx → ASSE Y veicolo
        - Vy → ASSE X veicolo
        - Z → X × Y
        """
        lights2 = self._ensure_numpy_array(lights_frame2, (2, 2))
        
        dir_Vx_cam = self.K_inv @ np.append(Vx, 1.0)
        dir_Vx_cam /= np.linalg.norm(dir_Vx_cam)
        
        dir_Vy_cam = self.K_inv @ np.append(Vy, 1.0)
        dir_Vy_cam /= np.linalg.norm(dir_Vy_cam)
        
        X_veh_in_cam = dir_Vy_cam
        
        L2, R2 = lights2[0], lights2[1]
        L2_ray = self.K_inv @ np.append(L2, 1.0)
        R2_ray = self.K_inv @ np.append(R2, 1.0)
        
        segment = L2_ray - R2_ray
        Y_veh_in_cam = segment / np.linalg.norm(segment)
        
        Z_veh_in_cam = np.cross(X_veh_in_cam, Y_veh_in_cam)
        Z_veh_in_cam /= np.linalg.norm(Z_veh_in_cam)
        
        if Z_veh_in_cam[1] > 0:
            Z_veh_in_cam = -Z_veh_in_cam
        
        Y_veh_in_cam = np.cross(Z_veh_in_cam, X_veh_in_cam)
        Y_veh_in_cam /= np.linalg.norm(Y_veh_in_cam)
        
        R = np.column_stack([X_veh_in_cam, Y_veh_in_cam, Z_veh_in_cam])
        
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        
        offset_veh = np.array([self.lights_x_offset, 0.0, self.lights_height])
        offset_cam = R @ offset_veh
        
        origin_cam = center_3d - offset_cam
        
        tvec = origin_cam.reshape(3, 1).astype(np.float32)
        rvec, _ = cv2.Rodrigues(R)
        
        return rvec, tvec, R
    
    def estimate_pose(
        self,
        lights_frame1: Union[Tuple, np.ndarray],
        lights_frame2: Union[Tuple, np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Pipeline completa."""
        Vx, Vy = self.calculate_vanishing_points(lights_frame1, lights_frame2)
        if Vx is None or Vy is None:
            return None
        
        is_perp, dot = self.check_perpendicularity(Vx, Vy)
        motion_type = "TRANSLATION" if is_perp else "ROTATION"
        
        n_pi = self.compute_plane_pi(Vx, Vy)
        center_3d, dist = self.localize_light_segment_3d(lights_frame2, Vx, n_pi)
        
        rvec, tvec, R = self.reconstruct_pose_from_plane(
            lights_frame2, Vx, Vy, center_3d, dist
        )
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'Vx': Vx,
            'Vy': Vy,
            'n_pi': n_pi,
            'center_3d': center_3d,
            'distance': np.linalg.norm(tvec),
            'motion_type': motion_type,
            'is_perpendicular': is_perp,
            'dot_product': dot,
            'is_valid': True,
            'frame': frame_idx
        }
    
    def classify_motion_type(
        self,
        Vx,
        Vy,
        threshold: Optional[float] = None
    ) -> str:
        """Classifica tipo movimento."""
        is_perp, _ = self.check_perpendicularity(Vx, Vy, threshold)
        return "TRANSLATION" if is_perp else "ROTATION"
