"""
Tracking module - Tracking temporale dei fari e gestione re-detection.
"""

from .tracker import LightTracker
from .redetection import RedetectionManager

__all__ = [
    'LightTracker',
    'RedetectionManager',
]