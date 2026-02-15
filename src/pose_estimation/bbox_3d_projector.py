"""
BBox 3D Projector - AGGIORNATO PER NUOVO YAML

Proietta bounding box 3D del veicolo sull'immagine.
Usa 'outer' points come riferimento (coerente con VanishingPointSolver).
"""

import numpy as np
import cv2
from typing import Optional


class BBox3DProjector:
    """
    Proietta bounding box 3D del veicolo.
    
    COORDINATE (dal YAML):
    - Origine: centro asse ruote posteriori a livello suolo
    - X: avanti (positivo)
    - Y: sinistra (positivo)
    - Z: su (positivo)
    """
    
    def __init__(self, camera_params, vehicle_model: dict):
        self.camera_matrix = camera_params.camera_matrix
        self.dist_coeffs = camera_params.dist_coeffs
        
        vehicle_data = vehicle_model.get('vehicle', {})
        
        # ===== PARSING CORRETTO YAML (NUOVO FORMATO) =====
        tail_lights = vehicle_data.get('tail_lights', {})
        
        # Usa 'outer' come punto di riferimento (coerente con VanishingPointSolver)
        left_light_dict = tail_lights.get('left', {})
        right_light_dict = tail_lights.get('right', {})
        
        # Estrai outer points
        left_outer = left_light_dict.get('outer', [-0.30, 0.71, 1.04])
        right_outer = right_light_dict.get('outer', [-0.30, -0.71, 1.04])
        
        # Coordinate outer (reference per posa)
        self.lights_x = left_outer[0]      # -0.30m (posteriore)
        self.lights_y = left_outer[1]      # 0.71m (sinistra)
        self.lights_z = left_outer[2]      # 1.04m (altezza)
        
        # ===== DIMENSIONI BBOX =====
        dimensions = vehicle_data.get('dimensions', {})
        
        self.length = dimensions.get('length', 3.70)
        self.width = dimensions.get('width', 1.74)
        self.height = dimensions.get('height', 1.525)
        
        # ===== VERTICI BBOX 3D (HARDCODED - NON FIDARTI DEL YAML) =====
        # Base (suolo, Z=0)
        self.bottom_rear_left = np.array([-0.54, 0.87, 0.0], dtype=np.float32)
        self.bottom_rear_right = np.array([-0.54, -0.87, 0.0], dtype=np.float32)
        self.bottom_front_left = np.array([3.16, 0.87, 0.0], dtype=np.float32)
        self.bottom_front_right = np.array([3.16, -0.87, 0.0], dtype=np.float32)

        # Top (tetto, Z=1.525)
        self.top_rear_left = np.array([-0.54, 0.87, 1.525], dtype=np.float32)
        self.top_rear_right = np.array([-0.54, -0.87, 1.525], dtype=np.float32)
        self.top_front_left = np.array([3.16, 0.87, 1.525], dtype=np.float32)
        self.top_front_right = np.array([3.16, -0.87, 1.525], dtype=np.float32)
        
        # ==========================================================
        # BBOX VERTICES - SISTEMA VEICOLO
        # ==========================================================
        # La posa (rvec, tvec) ha origine alle RUOTE POSTERIORI
        # La bbox è già definita con origine alle ruote
        # NON serve offset
        # ==========================================================

        self.bbox_3d = np.array([
            self.bottom_rear_right,   # 0
            self.bottom_rear_left,    # 1
            self.bottom_front_left,   # 2
            self.bottom_front_right,  # 3
            self.top_rear_right,      # 4
            self.top_rear_left,       # 5
            self.top_front_left,      # 6
            self.top_front_right      # 7
        ], dtype=np.float32)

        print(f"[BBox3DProjector] Initialized:")
        print(f"  Vehicle: {self.length:.2f}m x {self.width:.2f}m x {self.height:.2f}m")
        print(f"  BBox origin: rear axle center at ground level (X-fwd, Y-left, Z-up)")
        print(f"  Outer ref: [{self.lights_x:.2f}, +/-{self.lights_y:.2f}, {self.lights_z:.2f}]")
    
    def project_bbox(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Proietta bbox 3D sull'immagine.
        
        Args:
            rvec: vettore rotazione (3,1)
            tvec: vettore traslazione (3,1)
        
        Returns:
            np.ndarray (8, 2) con coordinate 2D dei vertici, o None se errore
        """
        try:
            # Proietta 8 vertici 3D → 2D
            points_2d, _ = cv2.projectPoints(
                self.bbox_3d,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # Reshape a (8, 2)
            points_2d = points_2d.reshape(-1, 2)
            
            return points_2d
        
        except Exception as e:
            print(f"  ⚠️ BBox projection failed: {e}")
            return None
    
    def get_bbox_vertices_3d(self) -> np.ndarray:
        """Ritorna vertici bbox 3D nel sistema veicolo."""
        return self.bbox_3d.copy()