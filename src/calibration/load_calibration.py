"""
Load Calibration - Caricamento parametri di calibrazione camera.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraParameters:
    """
    Container per i parametri della camera.
    
    Attributes:
        camera_matrix: Matrice intrinseca K (3x3)
        dist_coeffs: Coefficienti di distorsione (5,) o (1, 5)
        resolution: Tuple (width, height) della risoluzione
        fps: Frame rate del video
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    resolution: Optional[tuple] = None
    fps: Optional[float] = None
    
    def __post_init__(self):
        """Validazione parametri dopo inizializzazione."""
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix deve essere 3x3, ricevuto: {self.camera_matrix.shape}")
        
        # Assicurati che dist_coeffs sia (5,) o (1, 5)
        if self.dist_coeffs.ndim == 1:
            if len(self.dist_coeffs) < 5:
                # Padding con zeri se necessario
                self.dist_coeffs = np.pad(self.dist_coeffs, (0, 5 - len(self.dist_coeffs)))
        elif self.dist_coeffs.ndim == 2:
            self.dist_coeffs = self.dist_coeffs.flatten()[:5]
    
    @property
    def fx(self) -> float:
        """Lunghezza focale X."""
        return self.camera_matrix[0, 0]
    
    @property
    def fy(self) -> float:
        """Lunghezza focale Y."""
        return self.camera_matrix[1, 1]
    
    @property
    def cx(self) -> float:
        """Centro ottico X."""
        return self.camera_matrix[0, 2]
    
    @property
    def cy(self) -> float:
        """Centro ottico Y."""
        return self.camera_matrix[1, 2]


def load_camera_calibration(calibration_path: str, 
                            resolution: Optional[tuple] = None,
                            fps: Optional[float] = None) -> CameraParameters:
    """
    Carica i parametri di calibrazione della camera da file .npy.
    
    Il file .npy deve contenere un dizionario o array con:
    - 'camera_matrix' o 'mtx': Matrice intrinseca K (3x3)
    - 'dist_coeffs' o 'dist': Coefficienti di distorsione (5,)
    
    Oppure un array (2,) dove:
    - arr[0] = camera_matrix
    - arr[1] = dist_coeffs
    
    Args:
        calibration_path: Path al file .npy
        resolution: Risoluzione (width, height) opzionale
        fps: Frame rate opzionale
        
    Returns:
        CameraParameters object
        
    Raises:
        FileNotFoundError: Se il file non esiste
        ValueError: Se il formato non è valido
    """
    calib_file = Path(calibration_path)
    
    if not calib_file.exists():
        raise FileNotFoundError(f"File di calibrazione non trovato: {calibration_path}")
    
    # Carica il file
    data = np.load(calibration_path, allow_pickle=True)
    
    camera_matrix = None
    dist_coeffs = None
    
    # Caso 1: Dizionario con chiavi esplicite
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()  # Converti a dict
    
    if isinstance(data, dict):
        # Cerca camera_matrix
        if 'camera_matrix' in data:
            camera_matrix = data['camera_matrix']
        elif 'mtx' in data:
            camera_matrix = data['mtx']
        elif 'K' in data:
            camera_matrix = data['K']
        
        # Cerca dist_coeffs
        if 'dist_coeffs' in data:
            dist_coeffs = data['dist_coeffs']
        elif 'dist' in data:
            dist_coeffs = data['dist']
        elif 'distortion' in data:
            dist_coeffs = data['distortion']
    
    # Caso 2: Array (2,) con [camera_matrix, dist_coeffs]
    elif isinstance(data, np.ndarray) and len(data) == 2:
        camera_matrix = data[0]
        dist_coeffs = data[1]
    
    # Caso 3: File npz
    elif hasattr(data, 'files'):
        files = data.files
        if 'camera_matrix' in files:
            camera_matrix = data['camera_matrix']
        elif 'mtx' in files:
            camera_matrix = data['mtx']
        
        if 'dist_coeffs' in files:
            dist_coeffs = data['dist_coeffs']
        elif 'dist' in files:
            dist_coeffs = data['dist']
    
    # Validazione
    if camera_matrix is None:
        raise ValueError(
            f"Impossibile trovare 'camera_matrix' nel file {calibration_path}. "
            f"Chiavi trovate: {data.keys() if isinstance(data, dict) else 'array'}"
        )
    
    if dist_coeffs is None:
        print(f"Warning: Coefficienti di distorsione non trovati, usando zeri")
        dist_coeffs = np.zeros(5)
    
    return CameraParameters(
        camera_matrix=camera_matrix.astype(np.float64),
        dist_coeffs=dist_coeffs.astype(np.float64),
        resolution=resolution,
        fps=fps
    )


def load_camera_from_config(config: dict) -> CameraParameters:
    """
    Carica i parametri camera da un dizionario di configurazione.
    
    Args:
        config: Dizionario di configurazione (da camera_config.yaml)
        
    Returns:
        CameraParameters object
    """
    camera_cfg = config.get('camera', {})
    
    # Prova a caricare da file
    calib_file = camera_cfg.get('calibration_file')
    if calib_file and Path(calib_file).exists():
        resolution = tuple(camera_cfg.get('resolution', {}).values()) or None
        fps = camera_cfg.get('fps')
        return load_camera_calibration(calib_file, resolution, fps)
    
    # Altrimenti costruisci da parametri in config
    intrinsics = camera_cfg.get('intrinsics', {})
    distortion = camera_cfg.get('distortion', {})
    
    camera_matrix = np.array([
        [intrinsics.get('fx', 800.0), 0, intrinsics.get('cx', 640.0)],
        [0, intrinsics.get('fy', 800.0), intrinsics.get('cy', 360.0)],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.array([
        distortion.get('k1', 0.0),
        distortion.get('k2', 0.0),
        distortion.get('p1', 0.0),
        distortion.get('p2', 0.0),
        distortion.get('k3', 0.0)
    ], dtype=np.float64)
    
    resolution_cfg = camera_cfg.get('resolution', {})
    resolution = (resolution_cfg.get('width', 1280), resolution_cfg.get('height', 720))
    fps = camera_cfg.get('fps', 30)
    
    return CameraParameters(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        resolution=resolution,
        fps=fps
    )

def load_camera_calibration_simple(calibration_file):
    """
    Wrapper function that returns (camera_matrix, dist_coeffs) tuple
    for compatibility with existing code.
    
    Args:
        calibration_file: Path to calibration file
        
    Returns:
        Tuple (camera_matrix, dist_coeffs)
    """
    cam_params = load_camera_calibration(calibration_file)
    return cam_params.camera_matrix, cam_params.dist_coeffs

# Alias for backward compatibility
def load_camera_matrices(calibration_file):
    """Alias for load_camera_calibration_simple"""
    return load_camera_calibration_simple(calibration_file)
