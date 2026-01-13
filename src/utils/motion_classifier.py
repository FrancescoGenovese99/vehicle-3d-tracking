"""
Motion Classification Utility

Classifies vehicle motion as TRANSLATION or ROTATION
based on pose changes between frames.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class MotionClassifier:
    """
    Utility to classify vehicle motion type from pose data.
    
    Used across all tasks (Task 1, Task 2, PnP) to determine
    whether the vehicle is translating or rotating.
    """
    
    def __init__(
        self,
        yaw_translation_threshold: float = 5.0,  # degrees
        yaw_rotation_threshold: float = 15.0     # degrees
    ):
        """
        Initialize classifier.
        
        Args:
            yaw_translation_threshold: Max yaw change for pure translation (degrees)
            yaw_rotation_threshold: Min yaw change for clear rotation (degrees)
        """
        self.yaw_trans_thresh = np.radians(yaw_translation_threshold)
        self.yaw_rot_thresh = np.radians(yaw_rotation_threshold)
        
        # History for tracking
        self.previous_yaw = None
    
    def extract_yaw_from_rvec(self, rvec: np.ndarray) -> float:
        """
        Extract yaw angle from rotation vector.
        
        Args:
            rvec: Rotation vector (3x1) from cv2.solvePnP or similar
            
        Returns:
            Yaw angle in radians
        """
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Extract yaw (rotation around Y-axis)
        # Using convention: yaw = atan2(R[0,2], R[0,0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return yaw
    
    def classify_from_pose_change(
        self,
        rvec_curr: np.ndarray,
        rvec_prev: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """
        Classify motion from pose change between frames.
        
        Args:
            rvec_curr: Current frame rotation vector
            rvec_prev: Previous frame rotation vector (optional)
            
        Returns:
            Tuple (motion_type, yaw_change) where:
            - motion_type: 'TRANSLATION', 'ROTATION', or 'MIXED'
            - yaw_change: Change in yaw angle (radians)
        """
        # Extract current yaw
        yaw_curr = self.extract_yaw_from_rvec(rvec_curr)
        
        if rvec_prev is None:
            # First frame: cannot determine, assume translation
            self.previous_yaw = yaw_curr
            return "TRANSLATION", 0.0
        
        # Extract previous yaw
        yaw_prev = self.extract_yaw_from_rvec(rvec_prev)
        
        # Calculate yaw change
        yaw_change = abs(yaw_curr - yaw_prev)
        
        # Normalize to [-π, π]
        if yaw_change > np.pi:
            yaw_change = 2 * np.pi - yaw_change
        
        # Classify
        if yaw_change < self.yaw_trans_thresh:
            motion_type = "TRANSLATION"
        elif yaw_change > self.yaw_rot_thresh:
            motion_type = "ROTATION"
        else:
            motion_type = "MIXED"
        
        # Update history
        self.previous_yaw = yaw_curr
        
        return motion_type, yaw_change
    
    def classify_from_absolute_yaw(
        self,
        rvec: np.ndarray,
        frontal_yaw_threshold: float = 10.0  # degrees
    ) -> str:
        """
        Classify based on absolute yaw angle.
        
        If vehicle is nearly frontal (yaw ≈ 0), likely translating straight.
        If vehicle is rotated significantly, likely turning.
        
        Args:
            rvec: Rotation vector
            frontal_yaw_threshold: Threshold for frontal alignment (degrees)
            
        Returns:
            'TRANSLATION' or 'ROTATION'
        """
        yaw = self.extract_yaw_from_rvec(rvec)
        yaw_deg = abs(np.degrees(yaw))
        
        if yaw_deg < frontal_yaw_threshold:
            return "TRANSLATION"
        else:
            return "ROTATION"
    
    def reset(self):
        """Reset motion history."""
        self.previous_yaw = None


def classify_motion_from_vanishing_point(
    segment_direction: np.ndarray,
    motion_direction: np.ndarray,
    perpendicularity_threshold: float = 0.2
) -> str:
    """
    Classify motion using vanishing point geometry.
    
    For Task 2: if light segment ⊥ motion direction → TRANSLATION
                otherwise → ROTATION
    
    Args:
        segment_direction: Direction vector of light segment (normalized)
        motion_direction: Direction vector from vanishing point (normalized)
        perpendicularity_threshold: Max |dot product| for perpendicularity
        
    Returns:
        'TRANSLATION' or 'ROTATION'
    """
    dot_product = abs(np.dot(segment_direction, motion_direction))
    
    if dot_product < perpendicularity_threshold:
        return "TRANSLATION"
    else:
        return "ROTATION"


def get_motion_color(motion_type: str) -> Tuple[int, int, int]:
    """
    Get BGR color for motion type visualization.
    
    Args:
        motion_type: 'TRANSLATION', 'ROTATION', or 'MIXED'
        
    Returns:
        BGR color tuple
    """
    colors = {
        "TRANSLATION": (0, 255, 0),    # Green
        "ROTATION": (0, 0, 255),       # Red
        "MIXED": (0, 165, 255)         # Orange
    }
    
    return colors.get(motion_type, (255, 255, 255))  # Default: white


if __name__ == "__main__":
    # Test
    classifier = MotionClassifier()
    
    # Simulate translation (no yaw change)
    rvec1 = np.array([[0.0], [0.0], [0.0]])
    rvec2 = np.array([[0.0], [0.01], [0.0]])  # Small change
    
    motion, yaw_change = classifier.classify_from_pose_change(rvec2, rvec1)
    print(f"Test 1: {motion}, yaw_change={np.degrees(yaw_change):.2f}°")
    
    # Simulate rotation
    rvec3 = np.array([[0.0], [0.3], [0.0]])  # Significant yaw
    motion, yaw_change = classifier.classify_from_pose_change(rvec3, rvec2)
    print(f"Test 2: {motion}, yaw_change={np.degrees(yaw_change):.2f}°")