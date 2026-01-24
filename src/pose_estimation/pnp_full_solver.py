"""
PnP Solver - Stima della posa 3D usando cv2.solvePnP.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from src.calibration.load_calibration import CameraParameters


class PnPSolver:
    """
    Risolve il problema Perspective-n-Point per stimare la posa 3D del veicolo.
    """
    
    def __init__(self, camera_params: CameraParameters, vehicle_config: Dict, pnp_config: Dict):
        """
        Inizializza il solver.
        
        Args:
            camera_params: Parametri della camera
            vehicle_config: Configurazione del veicolo (da vehicle_model.yaml)
            pnp_config: Configurazione PnP (da camera_config.yaml)
        """
        self.camera_matrix = camera_params.camera_matrix
        self.dist_coeffs = camera_params.dist_coeffs
        
        # Punti 3D del modello veicolo (nel sistema di riferimento del veicolo)
        self.object_points_3d = self._build_object_points(vehicle_config)
        
        # Configurazione PnP
        self.method = self._get_pnp_method(pnp_config.get('method', 'ITERATIVE'))
        self.use_extrinsic_guess = pnp_config.get('use_extrinsic_guess', False)
        self.refine_iterations = pnp_config.get('refine_iterations', 10)
        
        # RANSAC config
        ransac_cfg = pnp_config.get('ransac', {})
        self.use_ransac = ransac_cfg.get('enabled', False)
        self.ransac_reproj_error = ransac_cfg.get('reprojection_error', 8.0)
        self.ransac_confidence = ransac_cfg.get('confidence', 0.99)
        
        # Cache per posa precedente (se use_extrinsic_guess)
        self.previous_rvec = None
        self.previous_tvec = None
    
    def _build_object_points(self, vehicle_config: Dict) -> np.ndarray:
        vehicle_data = vehicle_config.get('vehicle', {})
        
        # Fari posteriori
        tail_lights = vehicle_data.get('tail_lights', {})
        points = [
            np.array(tail_lights.get('left', [-1.2, 0.7, 0.5])),
            np.array(tail_lights.get('right', [-1.2, -0.7, 0.5]))
        ]
        
        # AGGIUNGI: Angoli targa
        plate_corners = vehicle_data.get('license_plate_rear_corners', {})
        if plate_corners:
            points.extend([
                np.array(plate_corners.get('top_left', [0.0, 0.26, 0.455])),
                np.array(plate_corners.get('top_right', [0.0, -0.26, 0.455])),
                np.array(plate_corners.get('bottom_left', [0.0, 0.26, 0.345])),
                np.array(plate_corners.get('bottom_right', [0.0, -0.26, 0.345]))
            ])
        
        return np.array(points, dtype=np.float32)
        
        def _get_pnp_method(self, method_name: str) -> int:
            """
            Converte il nome del metodo in costante OpenCV.
            
            Args:
                method_name: Nome del metodo ('ITERATIVE', 'P3P', 'EPNP', ecc.)
                
            Returns:
                Costante OpenCV
            """
            methods = {
                'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
                'P3P': cv2.SOLVEPNP_P3P,
                'EPNP': cv2.SOLVEPNP_EPNP,
                'DLS': cv2.SOLVEPNP_DLS,
                'UPNP': cv2.SOLVEPNP_UPNP,
                'AP3P': cv2.SOLVEPNP_AP3P,
                'IPPE': cv2.SOLVEPNP_IPPE,
                'IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE,
                'SQPNP': cv2.SOLVEPNP_SQPNP
            }
            
            return methods.get(method_name.upper(), cv2.SOLVEPNP_ITERATIVE)
    
    def solve(self, image_points: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Risolve PnP per ottenere la posa del veicolo.
        
        Args:
            image_points: Punti 2D nell'immagine, array (N, 2) o (N, 1, 2)
                         Devono corrispondere all'ordine di self.object_points_3d
            
        Returns:
            Tuple (success, rvec, tvec) dove:
            - success: True se la soluzione è stata trovata
            - rvec: Vettore di rotazione (Rodrigues) shape (3, 1)
            - tvec: Vettore di traslazione shape (3, 1)
        """
        # Assicurati che image_points abbia shape corretta
        if image_points.shape[0] != self.object_points_3d.shape[0]:
            print(f"Errore: numero di punti non corrisponde. "
                  f"Attesi {self.object_points_3d.shape[0]}, ricevuti {image_points.shape[0]}")
            return False, None, None
        
        # Reshape se necessario
        if image_points.ndim == 2 and image_points.shape[1] == 2:
            image_points = image_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Prepara parametri per solvePnP
        object_points = self.object_points_3d.reshape(-1, 1, 3)
        
        # Initial guess (se disponibile)
        use_guess = self.use_extrinsic_guess and \
                   self.previous_rvec is not None and \
                   self.previous_tvec is not None
        
        try:
            if self.use_ransac:
                # SolvePnP con RANSAC
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    reprojectionError=self.ransac_reproj_error,
                    confidence=self.ransac_confidence,
                    flags=self.method
                )
                
                if success and inliers is not None:
                    print(f"   PnP RANSAC: {len(inliers)}/{len(object_points)} inliers")
            else:
                # SolvePnP standard
                if use_guess:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec=self.previous_rvec,
                        tvec=self.previous_tvec,
                        useExtrinsicGuess=True,
                        flags=self.method
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=self.method
                    )
            
            if success:
                # Opzionale: refine con iterazioni aggiuntive
                if self.refine_iterations > 0:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec
                    )
                
                # Salva per prossima iterazione
                self.previous_rvec = rvec.copy()
                self.previous_tvec = tvec.copy()
                
                return True, rvec, tvec
            else:
                return False, None, None
                
        except cv2.error as e:
            print(f"Errore OpenCV in solvePnP: {e}")
            return False, None, None
    
    def compute_reprojection_error(self, image_points: np.ndarray, 
                                   rvec: np.ndarray, tvec: np.ndarray) -> float:
        """
        Calcola l'errore di riproiezione.
        
        Args:
            image_points: Punti 2D osservati
            rvec: Vettore di rotazione
            tvec: Vettore di traslazione
            
        Returns:
            Errore medio in pixel
        """
        # Proietta i punti 3D
        projected_points, _ = cv2.projectPoints(
            self.object_points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Reshape
        projected_points = projected_points.reshape(-1, 2)
        if image_points.ndim == 3:
            image_points = image_points.reshape(-1, 2)
        
        # Calcola errore
        error = np.linalg.norm(image_points - projected_points, axis=1).mean()
        
        return error
    
    def rvec_to_rotation_matrix(self, rvec: np.ndarray) -> np.ndarray:
        """
        Converte vettore di rotazione (Rodrigues) in matrice di rotazione.
        
        Args:
            rvec: Vettore di rotazione (3, 1)
            
        Returns:
            Matrice di rotazione (3, 3)
        """
        R, _ = cv2.Rodrigues(rvec)
        return R
    
    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Converte matrice di rotazione in angoli di Eulero (roll, pitch, yaw).
        
        Args:
            R: Matrice di rotazione (3, 3)
            
        Returns:
            Tuple (roll, pitch, yaw) in radianti
        """
        # Estrai angoli di Eulero (convenzione XYZ)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def get_pose_info(self, rvec: np.ndarray, tvec: np.ndarray) -> Dict:
        """
        Estrae informazioni leggibili dalla posa.
        
        Args:
            rvec: Vettore di rotazione
            tvec: Vettore di traslazione
            
        Returns:
            Dizionario con informazioni sulla posa
        """
        R = self.rvec_to_rotation_matrix(rvec)
        roll, pitch, yaw = self.rotation_matrix_to_euler(R)
        
        # Distanza dal veicolo
        distance = np.linalg.norm(tvec)
        
        info = {
            'translation': {
                'x': float(tvec[0, 0]),
                'y': float(tvec[1, 0]),
                'z': float(tvec[2, 0]),
                'distance': float(distance)
            },
            'rotation': {
                'roll': float(np.degrees(roll)),
                'pitch': float(np.degrees(pitch)),
                'yaw': float(np.degrees(yaw))
            },
            'rotation_matrix': R.tolist(),
            'rvec': rvec.flatten().tolist(),
            'tvec': tvec.flatten().tolist()
        }
        
        return info
    
    def reset(self):
        """Reset della cache per extrinsic guess."""
        self.previous_rvec = None
        self.previous_tvec = None