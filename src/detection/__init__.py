"""
Detection modules for vehicle features.
"""

from .light_detector import LightDetector
from .plate_detector import PlateDetector
from .candidate_selector import CandidateSelector

__all__ = [
    'LightDetector',
    'PlateDetector',
    'CandidateSelector'
]