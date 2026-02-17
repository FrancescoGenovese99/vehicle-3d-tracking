"""
load_calibration.py

Utilities for loading camera intrinsic parameters from calibration files.

Supported file formats
----------------------
.npy  – NumPy array saved with ``np.save``.  Interpreted as either:
            • A pickled dict with keys 'camera_matrix' / 'mtx' / 'K'
              and 'dist_coeffs' / 'dist' / 'distortion'.
            • A length-2 array whose first element is the camera matrix
              and whose second is the distortion coefficients.
.npz  – NumPy archive saved with ``np.savez``.  Expected keys:
            'camera_matrix' (or 'mtx') and 'dist_coeffs' (or 'dist').

If distortion coefficients are absent a zero vector is used and a warning
is printed.  Coefficient arrays shorter than 5 elements are zero-padded to
length 5 to match the OpenCV convention.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


# ===========================================================================
# Data container
# ===========================================================================

@dataclass
class CameraParameters:
    """
    Immutable container for camera intrinsic parameters.

    Attributes:
        camera_matrix: 3×3 intrinsic matrix K.
        dist_coeffs:   Distortion coefficients, shape (5,) in OpenCV order
                       [k1, k2, p1, p2, k3].
        resolution:    Optional (width, height) in pixels.
        fps:           Optional frame rate.
    """
    camera_matrix: np.ndarray
    dist_coeffs:   np.ndarray
    resolution:    Optional[Tuple[int, int]] = None
    fps:           Optional[float]           = None

    def __post_init__(self):
        """Validate and normalise shapes after construction."""
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix must be (3, 3), got {self.camera_matrix.shape}"
            )

        # Ensure dist_coeffs is a flat vector of exactly 5 elements
        flat = self.dist_coeffs.flatten()
        if len(flat) < 5:
            flat = np.pad(flat, (0, 5 - len(flat)))
        self.dist_coeffs = flat[:5]

    # Convenience accessors for the principal camera parameters

    @property
    def fx(self) -> float:
        """Focal length along the x-axis (pixels)."""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length along the y-axis (pixels)."""
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        """Principal point x-coordinate (pixels)."""
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        """Principal point y-coordinate (pixels)."""
        return float(self.camera_matrix[1, 2])


# ===========================================================================
# Core loading function
# ===========================================================================

def load_camera_calibration(
    calibration_path: str,
    resolution: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
) -> CameraParameters:
    """
    Load camera intrinsic parameters from a .npy or .npz calibration file.

    The function tries three interpretations in order:
        1. Pickled dict  – keys 'camera_matrix'/'mtx'/'K' and
                           'dist_coeffs'/'dist'/'distortion'.
        2. Length-2 array – arr[0] = camera matrix, arr[1] = dist coeffs.
        3. .npz archive  – same key names as case 1.

    Args:
        calibration_path: Path to the calibration file (.npy or .npz).
        resolution:       Optional (width, height) to store in the returned object.
        fps:              Optional frame rate to store in the returned object.

    Returns:
        CameraParameters instance with float64 arrays.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
        ValueError:        If the file format is not recognised or the camera
                           matrix cannot be located.
    """
    calib_file = Path(calibration_path)
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    data = np.load(calibration_path, allow_pickle=True)

    camera_matrix = None
    dist_coeffs   = None

    # --- Case 1: pickled dict (saved with np.save on a dict object) ---
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()  # unwrap the 0-d object array to recover the dict

    if isinstance(data, dict):
        camera_matrix = (data.get('camera_matrix')
                         or data.get('mtx')
                         or data.get('K'))
        dist_coeffs   = (data.get('dist_coeffs')
                         or data.get('dist')
                         or data.get('distortion'))

    # --- Case 2: length-2 array [camera_matrix, dist_coeffs] ---
    elif isinstance(data, np.ndarray) and len(data) == 2:
        camera_matrix = data[0]
        dist_coeffs   = data[1]

    # --- Case 3: .npz archive ---
    elif hasattr(data, 'files'):
        camera_matrix = (data['camera_matrix'] if 'camera_matrix' in data.files
                         else data.get('mtx'))
        dist_coeffs   = (data['dist_coeffs'] if 'dist_coeffs' in data.files
                         else data.get('dist'))

    if camera_matrix is None:
        keys = list(data.keys()) if isinstance(data, dict) else getattr(data, 'files', 'N/A')
        raise ValueError(
            f"Could not locate 'camera_matrix' in {calibration_path}. "
            f"Keys found: {keys}"
        )

    if dist_coeffs is None:
        print(f"Warning: distortion coefficients not found in {calibration_path} "
              f"– using zeros.")
        dist_coeffs = np.zeros(5)

    return CameraParameters(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist_coeffs,   dtype=np.float64),
        resolution=resolution,
        fps=fps,
    )


# ===========================================================================
# Config-driven loader
# ===========================================================================

def load_camera_from_config(config: dict) -> CameraParameters:
    """
    Build a CameraParameters instance from a configuration dictionary.

    If a calibration file is referenced in the config and exists on disk,
    it is loaded via ``load_camera_calibration``.  Otherwise the function
    falls back to constructing the intrinsic matrix from individual
    parameters (fx, fy, cx, cy) and distortion coefficients listed in the
    config, which is useful during development before a physical calibration
    is available.

    Args:
        config: Top-level configuration dict (from camera_config.yaml).

    Returns:
        CameraParameters instance.
    """
    camera_cfg = config.get('camera', {})

    calib_file = camera_cfg.get('calibration_file')
    if calib_file and Path(calib_file).exists():
        res_cfg    = camera_cfg.get('resolution', {})
        resolution = (res_cfg.get('width'), res_cfg.get('height')) if res_cfg else None
        return load_camera_calibration(calib_file, resolution, camera_cfg.get('fps'))

    # Fallback: assemble K from individual intrinsic parameters in config
    intr = camera_cfg.get('intrinsics', {})
    dist = camera_cfg.get('distortion', {})

    camera_matrix = np.array([
        [intr.get('fx', 800.0), 0.0,                intr.get('cx', 640.0)],
        [0.0,                   intr.get('fy', 800.0), intr.get('cy', 360.0)],
        [0.0,                   0.0,                1.0],
    ], dtype=np.float64)

    dist_coeffs = np.array([
        dist.get('k1', 0.0),
        dist.get('k2', 0.0),
        dist.get('p1', 0.0),
        dist.get('p2', 0.0),
        dist.get('k3', 0.0),
    ], dtype=np.float64)

    res_cfg    = camera_cfg.get('resolution', {})
    resolution = (res_cfg.get('width', 1280), res_cfg.get('height', 720))

    return CameraParameters(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        resolution=resolution,
        fps=camera_cfg.get('fps', 30),
    )


# ===========================================================================
# Convenience wrappers (backward compatibility)
# ===========================================================================

def load_camera_calibration_simple(
    calibration_file: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper that unpacks CameraParameters into a plain tuple.

    Retained for compatibility with call sites that expect the older
    ``(camera_matrix, dist_coeffs)`` return convention.

    Args:
        calibration_file: Path to the calibration file.

    Returns:
        (camera_matrix, dist_coeffs) as float64 NumPy arrays.
    """
    params = load_camera_calibration(calibration_file)
    return params.camera_matrix, params.dist_coeffs


def load_camera_matrices(calibration_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for ``load_camera_calibration_simple``."""
    return load_camera_calibration_simple(calibration_file)