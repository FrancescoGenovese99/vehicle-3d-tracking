"""
BBox 3D Projector - VERSIONE FINALE CORRETTA
- Posteriore ALLINEATO ai fari
- Validazione visiva con croci
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from src.calibration.load_calibration import CameraParameters


class BBox3DProjector:
    """Proietta bbox 3D allineata ESATTAMENTE ai fari."""
    
    def __init__(self, camera_params: CameraParameters, vehicle_config: Dict):
        self.camera_matrix = camera_params.camera_matrix
        self.dist_coeffs = camera_params.dist_coeffs
        
        vehicle_data = vehicle_config.get('vehicle', {})
        dimensions = vehicle_data.get('dimensions', {})
        tail_lights = vehicle_data.get('tail_lights', {})
        
        # Dimensioni veicolo
        self.length = dimensions.get('length', 3.70)
        self.width = dimensions.get('width', 1.74)
        self.height = dimensions.get('height', 1.525)
        
        # Posizione fari (coordinate 3D nel sistema veicolo)
        left_light = tail_lights.get('left', [-0.27, 0.70, 0.50])
        right_light = tail_lights.get('right', [-0.27, -0.70, 0.50])
        
        self.lights_x = left_light[0]      # -0.27m (posteriore)
        self.lights_z = left_light[2]       # 0.50m (altezza)
        self.lights_y_spacing = left_light[1] * 2  # 1.40m (distanza)
        
        # ✅ Vertici bbox + posizioni teoriche fari
        self.bbox_vertices_3d = self._compute_bbox_vertices()
        self.theoretical_lights_3d = np.array([left_light, right_light], dtype=np.float32)
        
        print(f"[BBox] Fari a: X={self.lights_x}m, Z={self.lights_z}m, spacing={self.lights_y_spacing}m")
    
    def _compute_bbox_vertices(self) -> np.ndarray:
        """
        Calcola vertici bbox.
        
        CHIAVE: Il posteriore inizia a X = lights_x (dove sono i fari)
                Il frontale finisce a X = lights_x + length
        """
        half_width = self.width / 2
        
        # Posteriore (dove stanno i fari)
        x_rear = self.lights_x  # -0.27m
        
        # Frontale
        x_front = x_rear + self.length  # -0.27 + 3.70 = 3.43m
        
        vertices = np.array([
            # Base (z=0, a terra)
            [x_rear, -half_width, 0.0],      # 0: rear-right
            [x_rear, half_width, 0.0],       # 1: rear-left
            [x_front, half_width, 0.0],      # 2: front-left
            [x_front, -half_width, 0.0],     # 3: front-right
            
            # Top (z=height)
            [x_rear, -half_width, self.height],   # 4
            [x_rear, half_width, self.height],    # 5
            [x_front, half_width, self.height],   # 6
            [x_front, -half_width, self.height]   # 7
        ], dtype=np.float32)
        
        return vertices
    
    def project_bbox(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Optional[np.ndarray]:
        """Proietta bbox."""
        try:
            projected, _ = cv2.projectPoints(
                self.bbox_vertices_3d,
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            return projected.reshape(-1, 2).astype(int)
        except:
            return None
    
    def project_theoretical_lights(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Proietta posizioni TEORICHE dei fari (dove dovrebbero stare sulla bbox).
        
        Returns:
            [[L_x, L_y], [R_x, R_y]] proiettati, o None
        """
        try:
            projected, _ = cv2.projectPoints(
                self.theoretical_lights_3d,
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            return projected.reshape(-1, 2).astype(int)
        except:
            return None
    
    def validate_bbox_alignment(
        self,
        measured_lights: np.ndarray,
        theoretical_lights: np.ndarray,
        threshold_pixels: float = 20.0
    ) -> Tuple[bool, float]:
        """
        Valida se bbox è ben allineata ai fari misurati.
        
        Args:
            measured_lights: Fari rilevati [[L_x, L_y], [R_x, R_y]]
            theoretical_lights: Fari teorici sulla bbox
            threshold_pixels: Errore massimo permesso
            
        Returns:
            (is_aligned, error_pixels)
        """
        # Calcola errore medio
        errors = np.linalg.norm(measured_lights - theoretical_lights, axis=1)
        mean_error = np.mean(errors)
        
        is_aligned = mean_error < threshold_pixels
        
        return is_aligned, mean_error
    
    def get_bbox_vertices_3d(self) -> np.ndarray:
        return self.bbox_vertices_3d.copy()