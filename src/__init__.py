"""
Vehicle 3D Tracking Package
Moduli per rilevamento, tracking e stima posa 3D di veicoli da fari posteriori.
"""

__version__ = "1.0.0"
__author__ = "Vehicle Tracking Team"

# Import principali per facilitare l'uso
from src.utils.config_loader import load_config
from src.calibration.load_calibration import load_camera_calibration

__all__ = [
    'load_config',
    'load_camera_calibration',
]