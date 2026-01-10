"""
BBox 3D Projector - Proiezione della bounding box 3D orientata sul frame.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from src.calibration.load_calibration import CameraParameters


class BBox3DProjector:
    """
    Proietta la bounding box 3D del veicolo sul piano immagine.
    """
    
    def __init__(self, camera_params: CameraParameters, vehicle_config: Dict):
        """
        Inizializza il projector.
        
        Args:
            camera_params: Parametri della camera
            vehicle_config: Configurazione del veicolo (da vehicle_model.yaml)
        """
        self.camera_matrix = camera_params.camera_matrix
        self.dist_coeffs = camera_params.dist_coeffs
        
        # Dimensioni veicolo
        vehicle_data = vehicle_config.get('vehicle', {})
        dimensions = vehicle_data.get('dimensions', {})
        
        self.length = dimensions.get('length', 5.0)
        self.width = dimensions.get('width', 3.0)
        self.height = dimensions.get('height', 1.7)
        
        # Vertici 3D della bbox nel sistema di riferimento del veicolo
        self.bbox_vertices_3d = self._compute_bbox_vertices()
    
    def _compute_bbox_vertices(self) -> np.ndarray:
        """
        Calcola i vertici 3D della bounding box nel sistema di riferimento del veicolo.
        
        Sistema di riferimento: origine al centro ruote posteriori a suolo
        - X: avanti (positivo verso fronte)
        - Y: sinistra (positivo verso sinistra)
        - Z: su (positivo verso l'alto)
        
        Returns:
            Array (8, 3) con i vertici della bbox
            Ordine: 4 vertici base (z=0) + 4 vertici top (z=height)
        """
        # Half dimensions
        half_width = self.width / 2
        
        # Vertici base (z = 0, a livello suolo)
        # Sistema: centro tra ruote posteriori, quindi posteriore = 0, frontale = +length
        vertices_base = np.array([
            [0, -half_width, 0],           # 0: posteriore-destra
            [0, half_width, 0],            # 1: posteriore-sinistra
            [self.length, half_width, 0],  # 2: frontale-sinistra
            [self.length, -half_width, 0], # 3: frontale-destra
        ], dtype=np.float32)
        
        # Vertici top (z = height)
        vertices_top = vertices_base.copy()
        vertices_top[:, 2] = self.height
        
        # Combina: [base, top]
        vertices = np.vstack([vertices_base, vertices_top])
        
        return vertices
    
    def project_bbox(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Proietta la bounding box 3D sul piano immagine.
        
        Args:
            rvec: Vettore di rotazione (3, 1)
            tvec: Vettore di traslazione (3, 1)
            
        Returns:
            Array (8, 2) con i punti 2D proiettati
        """
        # Proietta i vertici 3D
        projected_points, _ = cv2.projectPoints(
            self.bbox_vertices_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Reshape da (8, 1, 2) a (8, 2)
        projected_points = projected_points.reshape(-1, 2)
        
        return projected_points.astype(int)
    
    def draw_bbox_on_frame(self, frame: np.ndarray, 
                          projected_points: np.ndarray,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Disegna la bounding box 3D proiettata sul frame.
        
        Args:
            frame: Frame su cui disegnare
            projected_points: Punti 2D proiettati (8, 2)
            color: Colore BGR
            thickness: Spessore linee
            
        Returns:
            Frame con bbox disegnata
        """
        frame_copy = frame.copy()
        
        # Indici dei vertici: [0-3] base, [4-7] top
        # 0: post-dx, 1: post-sx, 2: front-sx, 3: front-dx
        # 4-7: corrispondenti top
        
        # Disegna base (z=0)
        base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in base_edges:
            pt1 = tuple(projected_points[i])
            pt2 = tuple(projected_points[j])
            cv2.line(frame_copy, pt1, pt2, color, thickness)
        
        # Disegna top (z=height)
        top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
        for i, j in top_edges:
            pt1 = tuple(projected_points[i])
            pt2 = tuple(projected_points[j])
            cv2.line(frame_copy, pt1, pt2, color, thickness)
        
        # Disegna pilastri verticali
        vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in vertical_edges:
            pt1 = tuple(projected_points[i])
            pt2 = tuple(projected_points[j])
            cv2.line(frame_copy, pt1, pt2, color, thickness)
        
        # Evidenzia il posteriore (dove sono i fari)
        # Linea più spessa sul lato posteriore
        rear_color = (0, 0, 255)  # Rosso per il posteriore
        cv2.line(frame_copy, tuple(projected_points[0]), tuple(projected_points[1]), 
                rear_color, thickness + 1)
        cv2.line(frame_copy, tuple(projected_points[4]), tuple(projected_points[5]), 
                rear_color, thickness + 1)
        
        return frame_copy
    
    def draw_axes(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                 axis_length: float = 1.0) -> np.ndarray:
        """
        Disegna gli assi del sistema di riferimento del veicolo.
        
        Args:
            frame: Frame su cui disegnare
            rvec: Vettore di rotazione
            tvec: Vettore di traslazione
            axis_length: Lunghezza degli assi in metri
            
        Returns:
            Frame con assi disegnati
        """
        frame_copy = frame.copy()
        
        # Definisci gli assi nel sistema di riferimento del veicolo
        axis_points = np.array([
            [0, 0, 0],                    # Origine
            [axis_length, 0, 0],          # Asse X (rosso - avanti)
            [0, axis_length, 0],          # Asse Y (verde - sinistra)
            [0, 0, axis_length]           # Asse Z (blu - su)
        ], dtype=np.float32)
        
        # Proietta gli assi
        projected_axes, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        projected_axes = projected_axes.reshape(-1, 2).astype(int)
        
        origin = tuple(projected_axes[0])
        x_end = tuple(projected_axes[1])
        y_end = tuple(projected_axes[2])
        z_end = tuple(projected_axes[3])
        
        # Disegna assi con colori diversi
        cv2.line(frame_copy, origin, x_end, (0, 0, 255), 3)    # X: Rosso
        cv2.line(frame_copy, origin, y_end, (0, 255, 0), 3)    # Y: Verde
        cv2.line(frame_copy, origin, z_end, (255, 0, 0), 3)    # Z: Blu
        
        # Etichette
        cv2.putText(frame_copy, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        cv2.putText(frame_copy, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2)
        
        return frame_copy
    
    def get_bbox_vertices_3d(self) -> np.ndarray:
        """
        Ottiene i vertici 3D della bbox.
        
        Returns:
            Array (8, 3) con i vertici
        """
        return self.bbox_vertices_3d.copy()
    
    def is_bbox_visible(self, projected_points: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Verifica se la bbox è visibile nel frame.
        
        Args:
            projected_points: Punti 2D proiettati
            frame_shape: Shape del frame (height, width)
            
        Returns:
            True se almeno un vertice è nel frame
        """
        h, w = frame_shape[:2]
        
        # Conta quanti punti sono dentro il frame
        visible_count = 0
        for point in projected_points:
            x, y = point
            if 0 <= x < w and 0 <= y < h:
                visible_count += 1
        
        # Considera visibile se almeno 2 vertici sono nel frame
        return visible_count >= 2
    
    def compute_bbox_area_2d(self, projected_points: np.ndarray) -> float:
        """
        Calcola l'area 2D approssimativa della bbox proiettata.
        
        Args:
            projected_points: Punti 2D proiettati (8, 2)
            
        Returns:
            Area in pixel^2
        """
        # Usa il convex hull dei punti proiettati
        hull = cv2.convexHull(projected_points.astype(np.int32))
        area = cv2.contourArea(hull)
        
        return area