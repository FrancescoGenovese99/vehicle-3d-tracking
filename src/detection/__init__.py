"""
Detection module - Rilevamento fari e selezione candidati.
"""

from .light_detector import LightDetector
from .candidate_selector import CandidateSelector, LightCandidate

__all__ = [
    'LightDetector',
    'CandidateSelector',
    'LightCandidate',
]