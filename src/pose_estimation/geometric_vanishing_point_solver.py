"""
Vanishing Point Solver - GEOMETRIC APPROACH
Calcola posa 3D usando:
1. Origine veicolo (già calcolata correttamente)
2. Vanishing Point → definisce direzione asse X (3D)
3. Targa → definisce direzione asse Y (3D)
4. Vincoli geometrici reali del veicolo per risolvere pitch/yaw
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Union, List
from collections import deque


class GeometricVanishingPointSolver:
    """Solver geometrico completo con VP e assi 3D."""
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        # ===== PARSING MODELLO VEICOLO =====
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
        
        # Dimensioni veicolo
        dimensions = vehicle_data.get('dimensions', {})
        self.vehicle_length = dimensions.get('length', 3.70)
        self.vehicle_width = dimensions.get('width', 1.74)
        self.vehicle_height = dimensions.get('height', 1.525)
        
        # Parametri aggiuntivi per la geometria
        self.plate_bottom_y = 0.0  # Posizione Y dei bottom della targa (metà larghezza)
        self.plate_height = 0.22   # Altezza tipica targa
        
        # ===== PARAMETRI ALGORITMO =====
        self.yaw_translation_threshold = np.radians(8)
        
        self.tti_min = 0.5
        self.tti_max = 30.0
        self.prev_distance = None
        
        # Storia per smoothing
        self.yaw_history = deque(maxlen=10)
        self.pitch_history = deque(maxlen=10)
        self.prev_yaw = 0.0
        self.prev_pitch = 0.0
        
        # Debug
        self.debug_mode = True
        
        print(f"[Geometric VP Solver] Inizializzato")
        print(f"  ✅ Geometria: VP + Targa + Assi 3D")
        print(f"  📏 Distanza fari: {self.lights_distance_real:.3f}m")
        print(f"  📐 Altezza fari: {self.lights_height:.3f}m")
    
    def _ensure_numpy_array(self, data: Union[Tuple, List, np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Converte input in numpy array della forma corretta."""
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
        """Stima distanza dalla larghezza apparente dei fari."""
        L, R = outer_points[0], outer_points[1]
        
        pixel_dist = np.linalg.norm(R - L)
        if pixel_dist < 1.0:
            return 10.0
        
        f_avg = (self.fx + self.fy) / 2.0
        Z = (self.lights_distance_real * f_avg) / pixel_dist
        
        return np.clip(Z, 2.0, 50.0)
    
    def compute_vehicle_origin(self, center_2d: np.ndarray, distance: float) -> np.ndarray:
        """Calcola origine veicolo nel camera frame (come nel codice originale)."""
        # Back-project centro fari
        center_ray = self.K_inv @ np.append(center_2d, 1.0)
        center_ray = center_ray / np.linalg.norm(center_ray)
        
        # Posizione 3D fari nel camera frame
        lights_pos_cam = center_ray * distance
        
        # Offset dai fari all'origine nel vehicle frame
        # Fari outer: X=-0.30, Y=0, Z=1.04
        # Origine: X=0, Y=0, Z=0
        lights_to_origin_veh = np.array([0.30, 0.0, -1.04], dtype=np.float32)
        
        return lights_pos_cam, lights_to_origin_veh
    
    def get_vanishing_point_3d_ray(self, vanishing_point_2d: np.ndarray) -> np.ndarray:
        """Converte vanishing point 2D in raggio 3D (direzione)."""
        vp_ray = self.K_inv @ np.append(vanishing_point_2d, 1.0)
        vp_ray = vp_ray / np.linalg.norm(vp_ray)
        return vp_ray
    
    def get_plate_y_axis_3d(self, plate_bottom: np.ndarray, distance: float) -> np.ndarray:
        """Calcola direzione asse Y dal bottom della targa (in 3D)."""
        BL, BR = plate_bottom[0], plate_bottom[1]
        
        # Back-project
        BL_ray = self.K_inv @ np.append(BL, 1.0)
        BR_ray = self.K_inv @ np.append(BR, 1.0)
        
        # Normalizza
        BL_ray = BL_ray / np.linalg.norm(BL_ray)
        BR_ray = BR_ray / np.linalg.norm(BR_ray)
        
        # Posizioni 3D approssimate (stessa profondità)
        BL_3d = BL_ray * distance
        BR_3d = BR_ray * distance
        
        # Direzione Y nel camera frame
        y_dir_cam = BR_3d - BL_3d
        y_dir_cam = y_dir_cam / np.linalg.norm(y_dir_cam)
        
        return y_dir_cam
    
    def solve_yaw_pitch_from_axes(self, x_axis_cam: np.ndarray, y_axis_cam: np.ndarray) -> Tuple[float, float]:
        """
        Risolve yaw e pitch dagli assi X e Y nel camera frame.
        
        Considerazioni:
        - Nel vehicle frame: X=[1,0,0], Y=[0,1,0], Z=[0,0,1]
        - Nel camera frame: X_cam = R * X_veh, Y_cam = R * Y_veh
        - Vogliamo trovare R (yaw + pitch) che meglio mappa gli assi
        """
        
        # Normalizza assi
        x_axis_cam = x_axis_cam / np.linalg.norm(x_axis_cam)
        y_axis_cam = y_axis_cam / np.linalg.norm(y_axis_cam)
        
        # Calcola asse Z ortogonale (prodotto vettoriale)
        z_axis_cam = np.cross(x_axis_cam, y_axis_cam)
        z_axis_cam = z_axis_cam / np.linalg.norm(z_axis_cam)
        
        # Ricostruisci matrice di rotazione
        R_cam_to_veh = np.column_stack([x_axis_cam, y_axis_cam, z_axis_cam])
        
        # Estrai angoli dalla matrice
        # Attenzione: potrebbe non essere perfettamente ortogonale per via della prospettiva
        
        # Metodo alternativo: risolvi per yaw e pitch che minimizzano l'errore
        # yaw = arctan2(X_cam[0], X_cam[2]) - π/2
        yaw = np.arctan2(x_axis_cam[0], x_axis_cam[2]) + np.pi/2
        
        # Pitch dalla componente Y dell'asse X (dovrebbe essere 0 in vehicle frame)
        # X_veh = [1,0,0], nel camera frame X_cam[1] = -sin(pitch)
        pitch = -np.arcsin(np.clip(x_axis_cam[1], -1.0, 1.0))
        
        # Normalizza yaw
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi
        
        return yaw, pitch
    
    def estimate_pose_with_vp_and_plate(
        self,
        outer_points: np.ndarray,
        plate_bottom: Optional[np.ndarray],
        vanishing_point: np.ndarray,
        frame_idx: int = 0
    ) -> Dict:
        """
        Stima posa usando geometria completa con VP e targa.
        
        Args:
            outer_points: Punti fari esterni [[x1,y1], [x2,y2]]
            plate_bottom: Punti bottom targa [[x1,y1], [x2,y2]] o None
            vanishing_point: Punto di fuga [x,y]
            frame_idx: Indice frame per debug
            
        Returns:
            Dizionario con risultati
        """
        
        # ===== 1. STIMA DISTANZA =====
        distance = self.estimate_distance_from_outer_points(outer_points)
        
        # ===== 2. CALCOLA ORIGINE VEICOLO =====
        center_2d = np.mean(outer_points, axis=0)
        lights_pos_cam, lights_to_origin_veh = self.compute_vehicle_origin(center_2d, distance)
        
        # ===== 3. DEFINISCI ASSI =====
        
        # Asse X: dalla direzione del vanishing point
        vp_ray = self.get_vanishing_point_3d_ray(vanishing_point)
        x_axis_cam = vp_ray.copy()
        
        # Asse Y: dalla targa se disponibile, altrimenti dai fari
        if plate_bottom is not None:
            y_axis_cam = self.get_plate_y_axis_3d(plate_bottom, distance)
            y_method = "plate"
        else:
            # Fallback: usa direzione tra fari come asse Y approssimato
            L, R = outer_points[0], outer_points[1]
            L_ray = self.K_inv @ np.append(L, 1.0)
            R_ray = self.K_inv @ np.append(R, 1.0)
            L_ray = L_ray / np.linalg.norm(L_ray)
            R_ray = R_ray / np.linalg.norm(R_ray)
            L_3d = L_ray * distance
            R_3d = R_ray * distance
            y_axis_cam = (R_3d - L_3d)
            y_axis_cam = y_axis_cam / np.linalg.norm(y_axis_cam)
            y_method = "lights"
        
        # ===== 4. RISOLVI YAW E PITCH =====
        yaw_raw, pitch_raw = self.solve_yaw_pitch_from_axes(x_axis_cam, y_axis_cam)
        
        # ===== 5. SMOOTHING TEMPORALE =====
        self.yaw_history.append(yaw_raw)
        self.pitch_history.append(pitch_raw)
        
        yaw = np.median(self.yaw_history)
        pitch = np.median(self.pitch_history)
        
        # Limita variazioni brusche
        max_delta = np.radians(15)
        if len(self.yaw_history) > 1:
            prev_yaw_median = np.median(list(self.yaw_history)[:-1])
            delta_yaw = yaw - prev_yaw_median
            delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))  # Normalizza
            
            if abs(delta_yaw) > max_delta:
                yaw = prev_yaw_median + np.sign(delta_yaw) * max_delta
        
        self.prev_yaw = yaw
        self.prev_pitch = pitch
        
        # ===== 6. COSTRUISCI MATRICE DI ROTAZIONE =====
        # Matrice di rotazione: prima yaw (Z), poi pitch (Y)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        
        # Rotazione del veicolo (solo yaw nel suo frame)
        R_yaw = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Pitch camera (inclinazione)
        R_pitch = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  cp,  -sp],
            [0.0,  sp,   cp]
        ])
        
        # Cambio base: vehicle frame → camera frame
        R_veh_to_cam_base = np.array([
            [ 0.0, -1.0,  0.0],  # X_cam = -Y_veh
            [ 0.0,  0.0, -1.0],  # Y_cam = -Z_veh  
            [ 1.0,  0.0,  0.0],  # Z_cam = X_veh
        ])
        
        # Rotazione combinata: vehicle → camera
        R_obj_to_cam = R_pitch @ R_veh_to_cam_base @ R_yaw
        
        # ===== 7. CALCOLA POSIZIONE ORIGINE =====
        # Trasforma offset fari-origine in camera frame
        lights_to_origin_cam = R_obj_to_cam @ lights_to_origin_veh
        
        # Posizione origine nel camera frame
        origin_pos_cam = lights_pos_cam + lights_to_origin_cam
        
        # ===== 8. OUTPUT PER OPENCV =====
        rvec, _ = cv2.Rodrigues(R_obj_to_cam)
        tvec = origin_pos_cam.reshape(3, 1).astype(np.float32)
        
        # ===== 9. VALIDAZIONE =====
        tti = self.calculate_tti(distance, dt=1.0/30.0)
        tti_valid, tti_msg = self.validate_pose_with_tti(tti)
        
        # Debug info
        if self.debug_mode and frame_idx % 10 == 0:
            print(f"Frame {frame_idx}:")
            print(f"  📏 Distanza: {distance:.2f}m")
            print(f"  🧭 Yaw: {np.degrees(yaw):.1f}°, Pitch: {np.degrees(pitch):.1f}°")
            print(f"  📍 Origine: [{origin_pos_cam[0]:.2f}, {origin_pos_cam[1]:.2f}, {origin_pos_cam[2]:.2f}]")
            print(f"  🎯 Asse Y da: {y_method}")
            if tti is not None:
                print(f"  ⏱️  TTI: {tti:.1f}s ({tti_msg})")
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R_obj_to_cam,
            'distance': distance,
            'origin_3d': origin_pos_cam,
            'frame': frame_idx,
            'is_valid': True,
            'yaw': yaw,
            'pitch': pitch,
            'tti': tti,
            'tti_valid': tti_valid,
            'debug': {
                'method': 'geometric_vp',
                'yaw_degrees': np.degrees(yaw),
                'pitch_degrees': np.degrees(pitch),
                'has_plate': plate_bottom is not None,
                'y_method': y_method
            }
        }
    
    def calculate_tti(self, distance: float, dt: float) -> Optional[float]:
        """Calcola Time To Impact."""
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
        
        return np.clip(tti, -100.0, 100.0)
    
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
        """Reset TTI history."""
        self.prev_distance = None
    
    def reset_temporal_smoothing(self):
        """Reset smoothing history."""
        self.yaw_history.clear()
        self.pitch_history.clear()
    
    # Alias per backward compatibility
    def estimate_pose(
        self,
        lights_frame: Union[Tuple, np.ndarray],
        plate_bottom: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Backward compatibility - richiede anche vanishing point."""
        raise NotImplementedError("Questo solver richiede vanishing_point. Usa estimate_pose_with_vp_and_plate()")


# Classe legacy wrapper per compatibilità
class VanishingPointSolver(GeometricVanishingPointSolver):
    """Wrapper per compatibilità con codice esistente."""
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        super().__init__(camera_matrix, vehicle_model)
        print("[Legacy Wrapper] Usa GeometricVanishingPointSolver per nuove funzionalità")
    
    def estimate_pose(
        self,
        lights_frame: Union[Tuple, np.ndarray],
        plate_bottom: Optional[np.ndarray],
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """Legacy method - richiede vanishing point, non disponibile."""
        print("⚠️  Attenzione: questo solver richiede vanishing_point")
        print("   Usa estimate_pose_with_vp_and_plate() invece")
        return None