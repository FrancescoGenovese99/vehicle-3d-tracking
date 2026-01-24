"""
Camera Calibrator - Calibrazione camera da immagini scacchiera.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import glob


class CameraCalibrator:
    """
    Classe per calibrazione camera usando pattern a scacchiera.
    """
    
    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        Inizializza il calibratore.
        
        Args:
            pattern_size: Dimensione pattern (colonne, righe) degli angoli interni
            square_size: Dimensione di un quadrato in metri
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Prepara i punti 3D del pattern
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Liste per punti 3D e 2D
        self.objpoints = []  # Punti 3D nel mondo reale
        self.imgpoints = []  # Punti 2D nell'immagine
        self.image_size = None
    
    def find_corners(self, image: np.ndarray, visualize: bool = False) -> Optional[np.ndarray]:
        """
        Trova gli angoli della scacchiera in un'immagine.
        
        Args:
            image: Immagine BGR
            visualize: Se True, mostra l'immagine con i corner rilevati
            
        Returns:
            Array numpy con i corner, o None se non trovati
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Trova gli angoli
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret:
            # Refine dei corner con subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            if visualize:
                vis_img = image.copy()
                cv2.drawChessboardCorners(vis_img, self.pattern_size, corners, ret)
                cv2.imshow('Chessboard Corners', vis_img)
                cv2.waitKey(500)
            
            return corners
        
        return None
    
    def add_image(self, image_path: str, visualize: bool = False) -> bool:
        """
        Aggiungi un'immagine al set di calibrazione.
        
        Args:
            image_path: Path all'immagine
            visualize: Se True, visualizza i corner trovati
            
        Returns:
            True se i corner sono stati trovati, False altrimenti
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Errore: impossibile caricare {image_path}")
            return False
        
        if self.image_size is None:
            self.image_size = (img.shape[1], img.shape[0])
        
        corners = self.find_corners(img, visualize)
        
        if corners is not None:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            print(f"✓ Corner trovati in: {Path(image_path).name}")
            return True
        else:
            print(f"✗ Corner NON trovati in: {Path(image_path).name}")
            return False
    
    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Esegue la calibrazione con le immagini caricate.
        
        Returns:
            Tuple (camera_matrix, dist_coeffs, reprojection_error)
            
        Raises:
            ValueError: Se non ci sono abbastanza immagini
        """
        if len(self.objpoints) < 3:
            raise ValueError(f"Servono almeno 3 immagini per calibrare, trovate: {len(self.objpoints)}")
        
        print(f"\nCalibrazione con {len(self.objpoints)} immagini...")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            self.image_size,
            None, 
            None
        )
        
        # Calcola l'errore di riproiezione medio
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                rvecs[i], 
                tvecs[i], 
                camera_matrix, 
                dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        print(f"✓ Calibrazione completata!")
        print(f"  Errore di riproiezione medio: {mean_error:.4f} pixel")
        
        return camera_matrix, dist_coeffs, mean_error
    
    def save_calibration(self, output_path: str, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray):
        """
        Salva i parametri di calibrazione.
        
        Args:
            output_path: Path del file di output (.npy)
            camera_matrix: Matrice intrinseca
            dist_coeffs: Coefficienti di distorsione
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva come npz con chiavi esplicite
        np.savez(output_file, 
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 image_size=np.array(self.image_size))
        
        print(f"✓ Calibrazione salvata in: {output_path}")
    
    @staticmethod
    def calibrate_from_images(image_pattern: str, 
                             pattern_size: Tuple[int, int],
                             square_size: float,
                             output_path: str,
                             visualize: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Metodo statico per calibrare direttamente da un pattern di immagini.
        
        Args:
            image_pattern: Pattern glob per le immagini (es: "calib/*.jpg")
            pattern_size: Dimensione pattern (colonne, righe)
            square_size: Dimensione quadrato in metri
            output_path: Path dove salvare la calibrazione
            visualize: Se True, visualizza i corner trovati
            
        Returns:
            Tuple (camera_matrix, dist_coeffs, mean_error)
        """
        calibrator = CameraCalibrator(pattern_size, square_size)
        
        # Trova tutte le immagini
        image_files = glob.glob(image_pattern)
        if not image_files:
            raise FileNotFoundError(f"Nessuna immagine trovata con pattern: {image_pattern}")
        
        print(f"Trovate {len(image_files)} immagini per calibrazione")
        
        # Processa tutte le immagini
        for img_path in image_files:
            calibrator.add_image(img_path, visualize)
        
        # Calibra
        camera_matrix, dist_coeffs, mean_error = calibrator.calibrate()
        
        # Salva
        calibrator.save_calibration(output_path, camera_matrix, dist_coeffs)
        
        if visualize:
            cv2.destroyAllWindows()
        
        return camera_matrix, dist_coeffs, mean_error


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Rimuove la distorsione da un'immagine.
    
    Args:
        image: Immagine distorta
        camera_matrix: Matrice intrinseca camera
        dist_coeffs: Coefficienti di distorsione
        
    Returns:
        Immagine senza distorsione
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop l'immagine al ROI
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted