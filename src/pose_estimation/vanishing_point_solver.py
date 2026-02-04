"""
Vanishing Point Solver - VERSIONE FINALE CORRETTA
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Union, List
from collections import deque


class VanishingPointSolver:
    """VP Solver con GEOMETRIA CORRETTA."""
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        # ===== PARSING YAML =====
        vehicle_data = vehicle_model.get('vehicle', {})
        tail_lights = vehicle_data.get('tail_lights', {})
        
        left_dict = tail_lights.get('left', {})
        right_dict = tail_lights.get('right', {})
        
        self.left_top_3d = np.array(left_dict.get('top', [0.0, 0.51, 1.34]), dtype=np.float32)
        self.right_top_3d = np.array(right_dict.get('top', [0.0, -0.51, 1.34]), dtype=np.float32)
        
        self.left_outer_3d = np.array(left_dict.get('outer', [-0.30, 0.71, 1.04]), dtype=np.float32)
        self.right_outer_3d = np.array(right_dict.get('outer', [-0.30, -0.71, 1.04]), dtype=np.float32)
        
        self.top_outer_distance_real = np.linalg.norm(self.left_outer_3d - self.left_top_3d)
        self.lights_distance_real = tail_lights.get('distance_between_outer_points', 1.42)
        
        self.lights_height = self.left_outer_3d[2]  # 1.04m
        self.lights_x_offset = self.left_outer_3d[0]  # -0.30m
        
        # ===== DIMENSIONI VEICOLO =====
        dimensions = vehicle_data.get('dimensions', {})
        self.vehicle_length = dimensions.get('length', 3.70)
        self.vehicle_width = dimensions.get('width', 1.74)
        self.vehicle_height = dimensions.get('height', 1.525)
        
        # ===== PARAMETRI =====
        self.yaw_translation_threshold = np.radians(8)
        
        self.tti_min = 0.5
        self.tti_max = 30.0
        self.prev_distance = None
        
        self.vy_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=10)
        self.prev_yaw = 0.0
        
        self.reference_x_axis = None
        self.rotation_counter = 0
        self.translation_counter = 0
        self.rotation_angle_threshold = np.radians(12)
        self.persistence_frames = 3
        
        self.prev_center_3d = None
        
        # DEBUG MODE
        self.debug_mode = True
        
        print(f"[VP Solver] FINAL CORRECTED VERSION")
        print(f"  âœ… Yaw: CORRECTED formula")
        print(f"  âœ… Temporal smoothing: enabled")
        print(f"  ðŸ“ Outer-outer: {self.lights_distance_real:.3f}m")
    
    def _ensure_numpy_array(self, data: Union[Tuple, List, np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        if isinstance(data, (tuple, list)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Tipo non supportato: {type(data)}")
        
        if arr.shape != shape:
            arr = arr.reshape(shape)
        
        return arr
    
    def estimate_distance_from_outer_points(self, outer_points: np.ndarray) -> float:
        """Stima distanza da larghezza apparente fari."""
        L, R = outer_points[0], outer_points[1]
        
        pixel_dist = np.linalg.norm(R - L)
        if pixel_dist < 1.0:
            return 10.0
        
        f_avg = (self.fx + self.fy) / 2.0
        Z = (self.lights_distance_real * f_avg) / pixel_dist
        
        Z = max(2.0, min(50.0, Z))
        
        return Z
    
    def estimate_yaw_from_lights_geometry(
        self,
        outer_points: np.ndarray,
        distance: float
    ) -> float:
        """
        Stima yaw da geometria fari (fallback senza plate).
        """
        L_2d, R_2d = outer_points[0], outer_points[1]
        
        # Back-project in 3D
        L_ray = self.K_inv @ np.append(L_2d, 1.0)
        R_ray = self.K_inv @ np.append(R_2d, 1.0)
        
        # Normalizza raggi
        L_ray = L_ray / np.linalg.norm(L_ray)
        R_ray = R_ray / np.linalg.norm(R_ray)
        
        # Posizioni 3D nel camera frame
        L_3d = L_ray * distance
        R_3d = R_ray * distance
        
        # Direzione Y nel camera frame (Lâ†’R = -Y_veh)
        y_dir_cam_3d = R_3d - L_3d
        
        # Proietta su piano orizzontale
        y_horizontal = np.array([y_dir_cam_3d[0], 0.0, y_dir_cam_3d[2]])
        
        norm = np.linalg.norm(y_horizontal)
        if norm < 1e-6:
            return self.prev_yaw
        
        y_horizontal = y_horizontal / norm
        
        # Inverti per ottenere +Y_veh (punta verso sinistra)
        y_veh_horizontal = -y_horizontal
        
        # Angolo rispetto a Z_cam
        y_angle = np.arctan2(y_veh_horizontal[0], y_veh_horizontal[2])
        
        # Formula corretta
        yaw = y_angle + np.pi/2
        
        # Normalizza [-Ï€, Ï€]
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi
        
        return yaw
    
    def estimate_yaw_from_plate_bottom(
        self,
        plate_bottom: np.ndarray,
        distance: float
    ) -> float:
        """
        Stima yaw da plate bottom (metodo piÃ¹ accurato).
        """
        BL, BR = plate_bottom[0], plate_bottom[1]
        
        # Back-project
        BL_ray = self.K_inv @ np.append(BL, 1.0)
        BR_ray = self.K_inv @ np.append(BR, 1.0)
        
        # Normalizza
        BL_ray = BL_ray / np.linalg.norm(BL_ray)
        BR_ray = BR_ray / np.linalg.norm(BR_ray)
        
        # Posizioni 3D
        BL_3d = BL_ray * distance
        BR_3d = BR_ray * distance
        
        # Direzione Y
        y_dir_cam_3d = BR_3d - BL_3d
        
        # Proietta su piano orizzontale
        y_horizontal = np.array([y_dir_cam_3d[0], 0.0, y_dir_cam_3d[2]])
        
        norm = np.linalg.norm(y_horizontal)
        if norm < 1e-6:
            return self.prev_yaw
        
        y_horizontal = y_horizontal / norm
        
        # Inverti
        y_veh_horizontal = -y_horizontal
        
        # Angolo
        y_angle = np.arctan2(y_veh_horizontal[0], y_veh_horizontal[2])
        
        # Formula corretta
        yaw = y_angle + np.pi/2
        
        # Normalizza
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi
        
        return yaw
    
    def reconstruct_pose_robust(
        self,
        outer_points: np.ndarray,
        plate_bottom: Optional[np.ndarray],
        distance: float,
        frame_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ricostruzione posa CORRETTA per OpenCV.
        
        OpenCV convention:
        - rvec, tvec trasformano punti da OBJECT FRAME → CAMERA FRAME
        - R = R_obj_to_cam
        - tvec = posizione origine oggetto in camera frame
        """
        
        center_2d = np.mean(outer_points, axis=0)
        
        # ===== YAW (ADAPTIVE) =====
        if plate_bottom is not None and plate_bottom.shape == (2, 2):
            yaw_raw = self.estimate_yaw_from_plate_bottom(plate_bottom, distance)
            method = "plate_bottom"
        else:
            yaw_raw = self.estimate_yaw_from_lights_geometry(outer_points, distance)
            method = "lights_geometry"
        
        # ===== TEMPORAL SMOOTHING =====
        self.yaw_history.append(yaw_raw)
        yaw = np.median(self.yaw_history)
        
        # Limita variazioni brusche
        if len(self.yaw_history) > 1:
            max_delta = np.radians(15)
            prev_yaw_median = np.median(list(self.yaw_history)[:-1])
            delta = yaw - prev_yaw_median
            
            while delta > np.pi:
                delta -= 2 * np.pi
            while delta < -np.pi:
                delta += 2 * np.pi
            
            if abs(delta) > max_delta:
                yaw = prev_yaw_median + np.sign(delta) * max_delta
        
        self.prev_yaw = yaw
        
        if self.debug_mode and frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: yaw={np.degrees(yaw):.1f}°")
        
        # ===== STIMA PITCH CAMERA =====
        # Dalla posizione Y dei fari nell'immagine
        y_pixel_offset = center_2d[1] - self.cy
        pitch_cam = np.arctan2(y_pixel_offset, self.fy)
        
        if self.debug_mode and frame_idx % 10 == 0:
            print(f"  Pitch camera: {np.degrees(pitch_cam):.1f}°")
        
        # ===== COSTRUZIONE ROTAZIONE VEICOLO (nel suo frame) =====
        # Solo yaw attorno a Z verticale
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_veh = np.array([
            [ cy, -sy, 0.0],  # X_veh ruota nel piano XY
            [ sy,  cy, 0.0],  # Y_veh ruota nel piano XY
            [0.0, 0.0, 1.0]   # Z_veh rimane verticale
        ], dtype=np.float32)
        
        # ===== ROTAZIONE PITCH CAMERA =====
        cp, sp = np.cos(pitch_cam), np.sin(pitch_cam)
        R_pitch_cam = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  cp,  -sp],
            [0.0,  sp,   cp]
        ], dtype=np.float32)
        
        # ===== TRASFORMAZIONE VEHICLE → CAMERA =====
        # Convenzioni:
        # VEHICLE: X avanti, Y sinistra, Z su
        # CAMERA: X destra, Y giù, Z profondità (uscente)
        
        # Matrice cambio base (SENZA pitch camera)
        R_veh_to_cam_base = np.array([
            [ 0.0, -1.0,  0.0],  # X_cam = -Y_veh
            [ 0.0,  0.0, -1.0],  # Y_cam = -Z_veh  
            [ 1.0,  0.0,  0.0],  # Z_cam = X_veh
        ], dtype=np.float32)
        
        # Combina: pitch camera * cambio base * rotazione veicolo
        R_obj_to_cam = R_pitch_cam @ R_veh_to_cam_base @ R_veh
        
        # ===== TRASLAZIONE (posizione origine veicolo in camera frame) =====
        
        # 1) Back-project centro fari
        center_ray = self.K_inv @ np.append(center_2d, 1.0)
        center_ray = center_ray / np.linalg.norm(center_ray)
        
        # 2) Posizione 3D fari nel camera frame
        lights_pos_cam = center_ray * distance
        
        # 3) Offset dai fari all'origine NEL FRAME VEICOLO
        # Fari outer: X=-0.30, Y=0, Z=1.04
        # Origine: X=0, Y=0, Z=0
        lights_to_origin_veh = np.array([0.30, 0.0, -1.04], dtype=np.float32)
        
        # 4) Trasforma offset in camera frame usando R_obj_to_cam
        lights_to_origin_cam = R_obj_to_cam @ lights_to_origin_veh
        
        # 5) Posizione origine veicolo in camera frame
        origin_pos_cam = lights_pos_cam + lights_to_origin_cam
        
        # ===== OUTPUT PER OPENCV =====
        rvec, _ = cv2.Rodrigues(R_obj_to_cam)
        tvec = origin_pos_cam.reshape(3, 1).astype(np.float32)
        
        return rvec, tvec, R_obj_to_cam
                        
                
    
    def extract_yaw_from_rotation(self, R: np.ndarray) -> float:
        """Estrae yaw da matrice rotazione."""
        x_axis = R[:, 0].copy()
        x_axis[1] = 0.0
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            return 0.0
        x_axis /= norm
        
        yaw = np.arctan2(x_axis[0], x_axis[2])
        return yaw
    
    def classify_motion_type(self, R: np.ndarray) -> str:
        """Classifica tipo movimento."""
        x_axis = R[:, 2].copy()
        
        if self.reference_x_axis is None:
            self.reference_x_axis = x_axis / np.linalg.norm(x_axis)
            return "TRANSLATION"
        
        dot = np.clip(
            np.dot(self.reference_x_axis, x_axis) / np.linalg.norm(x_axis),
            -1.0, 1.0
        )
        angle = np.arccos(dot)
        
        if angle > self.rotation_angle_threshold:
            self.rotation_counter += 1
            self.translation_counter = 0
        else:
            self.translation_counter += 1
            self.rotation_counter = 0
        
        if self.rotation_counter >= self.persistence_frames:
            return "ROTATION"
        
        return "TRANSLATION"
    
    def calculate_tti(self, distance: float, dt: float) -> Optional[float]:
        """Calcola TTI."""
        if self.prev_distance is None:
            self.prev_distance = distance
            return None
        
        V = (distance - self.prev_distance) / dt
        
        if abs(V) > 5.0:
            self.prev_distance = distance
            return None
        
        if V >= -0.01:
            self.prev_distance = distance
            return None
        
        tti = -distance / V
        self.prev_distance = distance
        
        return tti
    
    def validate_pose_with_tti(self, tti: Optional[float]) -> Tuple[bool, str]:
        """Valida posa usando TTI."""
        if tti is None:
            return True, "TTI not available"
        
        if tti < 0:
            if abs(tti) > self.tti_max:
                return False, f"Moving away too slowly (TTI={tti:.1f}s)"
            return True, f"Moving away (TTI={tti:.1f}s)"
        
        if tti < self.tti_min:
            return False, f"TTI too small (TTI={tti:.2f}s)"
        
        if tti > self.tti_max:
            return False, f"TTI too large (TTI={tti:.1f}s)"
        
        return True, f"TTI valid ({tti:.1f}s)"
    
    def reset_tti_history(self):
        """Reset TTI."""
        self.prev_distance = None
    
    def reset_temporal_smoothing(self):
        """Reset smoothing."""
        self.vy_history.clear()
        self.yaw_history.clear()
    
    def reset_vp_persistence(self):
        """Reset VP solver."""
        self.prev_center_3d = None
        self.prev_yaw = 0.0
        self.reference_x_axis = None
        self.rotation_counter = 0
        self.translation_counter = 0
        print("[VP Solver] Reset")
    
    def estimate_pose_multifeature(
        self,
        features_t2: Dict[str, np.ndarray],
        plate_bottom_t2: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Stima posa ROBUSTA."""
        outer_points = features_t2.get('outer')
        if outer_points is None:
            return None
        
        distance = self.estimate_distance_from_outer_points(outer_points)
        
        rvec, tvec, R = self.reconstruct_pose_robust(
            outer_points,
            plate_bottom_t2,
            distance,
            frame_idx
        )
        
        motion_type = self.classify_motion_type(R)
        
        tti = self.calculate_tti(distance, dt=1.0/30.0)
        tti_valid, tti_msg = self.validate_pose_with_tti(tti)
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'distance': distance,
            'frame': frame_idx,
            'is_valid': True,
            'motion_type': motion_type,
            'tti': tti,
            'tti_valid': tti_valid,
            'debug': {
                'method': 'robust_yaw',
                'yaw_degrees': np.degrees(self.prev_yaw),
                'pitch_degrees': 0.0,
                'has_plate': plate_bottom_t2 is not None
            }
        }
    
    def estimate_pose(
        self,
        lights_frame: Union[Tuple, np.ndarray],
        plate_bottom: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Backward compatibility."""
        features = {'outer': self._ensure_numpy_array(lights_frame, (2, 2))}
        return self.estimate_pose_multifeature(features, plate_bottom, frame_idx)