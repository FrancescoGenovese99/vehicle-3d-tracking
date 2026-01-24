"""
Pose estimation modules for all three tasks.
"""

# Task 1: Homography
from .homography_solver import HomographySolver

# Task 2: Vanishing Point
from .vanishing_point_solver import VanishingPointSolver

# Task 3: Symmetry (SKIP - non implementato)
# from .symmetry_solver import SymmetrySolver

# Metodo PnP (confronto/opzionale)
from .pnp_full_solver import PnPSolver

# Utility condivise
from .bbox_3d_projector import BBox3DProjector

# NOTE: translational_solver.py deprecato - sostituito da vanishing_point_solver.py

__all__ = [
    'HomographySolver',      # Task 1
    'VanishingPointSolver',  # Task 2
    'PnPSolver',             # PnP comparison
    'BBox3DProjector'        # Shared utility
]