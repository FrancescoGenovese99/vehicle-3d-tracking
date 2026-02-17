"""
Pose estimation modules.
"""

from .pnp_pose_estimator import PnPPoseEstimator
from .bbox_3d_projector import BBox3DProjector
from .vp_pose_estimator import VPPoseEstimator

VanishingPointSolver = PnPPoseEstimator

__all__ = [
    'VanishingPointSolver',
    'BBox3DProjector',
]