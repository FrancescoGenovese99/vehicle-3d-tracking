"""
Task 2: Vanishing Point Pose Estimation

Estimates vehicle pose using vanishing point calculated from
tail light trajectories across two consecutive frames.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple


class VanishingPointSolver:
    """
    Solves vehicle pose using vanishing point method.
    
    Method (as per Task 2 specifications):
    1. Detect tail lights in frame N: L1', R1'
    2. Track lights to frame N+1: L2', R2'
    3. Calculate vanishing point Vx = intersection(L1'L2', R1'R2')
    4. Convert Vx to 3D motion direction via K⁻¹
    5. Verify perpendicularity: light_segment ⊥ motion_direction
    6. Estimate distance to plane π using metric constraint (light distance = 1.40m)
    7. Reconstruct full pose [R | t]
    """
    
    def __init__(self, camera_matrix: np.ndarray, vehicle_model: dict):
        """
        Initialize solver.
        
        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            vehicle_model: Dictionary with vehicle geometry
        """
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        # Extract vehicle parameters
        self.tail_lights_3d = np.array([
            vehicle_model['vehicle']['tail_lights']['left'],
            vehicle_model['vehicle']['tail_lights']['right']
        ], dtype=np.float32)
        
        self.lights_distance_real = vehicle_model['vehicle']['tail_lights']['distance_between']
        
        # Camera parameters
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
    def calculate_vanishing_point(
        self,
        lights_prev: np.ndarray,
        lights_curr: np.ndarray,
        tolerance: float = 10.0
    ) -> Optional[np.ndarray]:
        """
        Calculate vanishing point from light trajectories.
        
        Args:
            lights_prev: Previous frame lights [[x1,y1], [x2,y2]] (2x2)
            lights_curr: Current frame lights [[x1,y1], [x2,y2]] (2x2)
            tolerance: Max pixel distance for valid intersection
            
        Returns:
            Vanishing point [u, v] or None if invalid
        """
        # Extract left and right light positions
        L1 = lights_prev[0]  # Left light frame N
        R1 = lights_prev[1]  # Right light frame N
        L2 = lights_curr[0]  # Left light frame N+1
        R2 = lights_curr[1]  # Right light frame N+1
        
        # Line 1: through L1 and L2
        # Line 2: through R1 and R2
        
        # Calculate vanishing point as intersection
        vp = self._line_intersection(L1, L2, R1, R2)
        
        if vp is None:
            return None
        
        # Validate: check if intersection is reasonable
        # (not too close to image borders, not at infinity)
        u, v = vp
        
        # Check if point is within reasonable bounds (extended image area)
        img_width = 2 * self.cx
        img_height = 2 * self.cy
        
        if abs(u) > img_width * 3 or abs(v) > img_height * 3:
            # Vanishing point too far (nearly parallel lines)
            return None
        
        return vp
    
    def _line_intersection(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Calculate intersection of two lines.
        
        Line 1: through p1 and p2
        Line 2: through p3 and p4
        
        Returns:
            Intersection point [x, y] or None if parallel
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            # Lines are parallel
            return None
        
        # Calculate intersection
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return np.array([x, y], dtype=np.float32)
    
    def vanishing_point_to_3d_direction(self, vp: np.ndarray) -> np.ndarray:
        """
        Convert vanishing point to 3D motion direction.
        
        Args:
            vp: Vanishing point [u, v] in pixels
            
        Returns:
            Normalized 3D direction vector
        """
        # Convert pixel coordinates to normalized image coordinates
        u, v = vp
        
        # Create homogeneous pixel coordinates
        pixel_homog = np.array([u, v, 1.0])
        
        # Convert to 3D direction: d = K⁻¹ · [u, v, 1]ᵀ
        direction_3d = self.K_inv @ pixel_homog
        
        # Normalize
        direction_normalized = direction_3d / np.linalg.norm(direction_3d)
        
        return direction_normalized
    
    def verify_perpendicularity(
        self,
        lights_curr: np.ndarray,
        motion_direction_3d: np.ndarray,
        threshold: float = 0.3
    ) -> bool:
        """
        Verify that light segment is perpendicular to motion direction.
        
        Args:
            lights_curr: Current frame lights [[x1,y1], [x2,y2]]
            motion_direction_3d: 3D motion direction vector
            threshold: Max absolute dot product for perpendicularity
            
        Returns:
            True if perpendicular (translational motion confirmed)
        """
        # Calculate light segment in image
        light_segment_2d = lights_curr[1] - lights_curr[0]  # R - L
        
        # Convert to 3D direction (approximate)
        # We use the direction in image plane
        segment_3d_homog = np.array([light_segment_2d[0], light_segment_2d[1], 0.0])
        segment_3d = self.K_inv @ np.append(segment_3d_homog[:2], 1.0)
        segment_3d_norm = segment_3d / np.linalg.norm(segment_3d)
        
        # Calculate dot product
        dot_product = np.abs(np.dot(segment_3d_norm, motion_direction_3d))
        
        # Should be close to 0 for perpendicularity
        is_perpendicular = dot_product < threshold
        
        return is_perpendicular
    
    def estimate_distance_to_plane(
        self,
        lights_curr: np.ndarray
    ) -> float:
        """
        Estimate distance to the plane π containing the lights.
        
        Uses the metric constraint: real distance between lights = 1.40m
        
        Args:
            lights_curr: Current frame lights [[x1,y1], [x2,y2]]
            
        Returns:
            Estimated distance Z in meters
        """
        # Calculate pixel distance between lights
        light_distance_pixels = np.linalg.norm(lights_curr[1] - lights_curr[0])
        
        # Use perspective projection relation:
        # pixel_distance = (focal_length * real_distance) / depth
        # => depth = (focal_length * real_distance) / pixel_distance
        
        # Use average focal length
        f_avg = (self.fx + self.fy) / 2.0
        
        # Calculate depth
        Z = (f_avg * self.lights_distance_real) / light_distance_pixels
        
        return Z
    
    def reconstruct_pose(
        self,
        lights_curr: np.ndarray,
        motion_direction_3d: np.ndarray,
        distance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct full vehicle pose [R | t].
        
        Args:
            lights_curr: Current frame lights [[x1,y1], [x2,y2]]
            motion_direction_3d: 3D motion direction (forward axis)
            distance: Estimated distance to vehicle
            
        Returns:
            (rvec, tvec, R) tuple
        """
        # Calculate vehicle center in 3D (midpoint of lights)
        light_center_2d = np.mean(lights_curr, axis=0)
        
        # Back-project light center to 3D at estimated distance
        # Convert pixel to normalized coordinates
        light_center_norm = self.K_inv @ np.array([light_center_2d[0], 
                                                     light_center_2d[1], 
                                                     1.0])
        light_center_norm = light_center_norm / light_center_norm[2]
        
        # Scale by distance to get 3D position
        # Position in camera frame
        X = light_center_norm[0] * distance
        Y = light_center_norm[1] * distance
        Z = distance
        
        tvec = np.array([[X], [Y], [Z]], dtype=np.float32)
        
        # Construct rotation matrix
        # Assume vehicle's forward direction aligns with motion direction
        # and use known geometry
        
        # Forward axis (Z-axis of vehicle) = motion direction
        z_axis = motion_direction_3d
        
        # Calculate vehicle's right axis (X-axis) from light positions
        # Direction from left to right light
        light_direction_2d = lights_curr[1] - lights_curr[0]
        light_direction_3d = self.K_inv @ np.array([light_direction_2d[0],
                                                      light_direction_2d[1],
                                                      0.0])
        x_axis = light_direction_3d / np.linalg.norm(light_direction_3d)
        
        # Up axis (Y-axis) = cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Recompute x_axis to ensure orthogonality
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Build rotation matrix
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to rvec
        rvec, _ = cv2.Rodrigues(R)
        
        return rvec, tvec, R
    
    def classify_motion_type(
        self,
        lights_curr: np.ndarray,
        motion_direction_3d: np.ndarray,
        dot_threshold: float = 0.2
    ) -> str:
        """
        Classify vehicle motion as TRANSLATION or ROTATION.
        
        Args:
            lights_curr: Current frame lights
            motion_direction_3d: 3D motion direction from vanishing point
            dot_threshold: Threshold for perpendicularity check
            
        Returns:
            'TRANSLATION' if pure translational motion
            'ROTATION' if turning/rotating motion
        """
        # Calculate light segment in image
        light_segment_2d = lights_curr[1] - lights_curr[0]
        
        # Convert to 3D direction
        segment_3d = self.K_inv @ np.append(light_segment_2d, 1.0)
        segment_3d_norm = segment_3d / np.linalg.norm(segment_3d)
        
        # Calculate dot product
        dot_product = np.abs(np.dot(segment_3d_norm, motion_direction_3d))
        
        # If perpendicular → TRANSLATION
        # If parallel → ROTATION
        if dot_product < dot_threshold:
            return "TRANSLATION"
        else:
            return "ROTATION"
    
    def estimate_pose(
        self,
        lights_prev: np.ndarray,
        lights_curr: np.ndarray,
        frame_idx: int = 0
    ) -> Optional[Dict]:
        """
        Full pose estimation pipeline.
        
        Args:
            lights_prev: Previous frame lights (2x2)
            lights_curr: Current frame lights (2x2)
            frame_idx: Current frame index (for logging)
            
        Returns:
            Dictionary with pose data or None if estimation failed
            {
                'rvec': rotation vector (3x1),
                'tvec': translation vector (3x1),
                'R': rotation matrix (3x3),
                'vanishing_point': [u,v] or None,
                'distance': float,
                'motion_type': 'TRANSLATION' or 'ROTATION',
                'is_valid': bool
            }
        """
        # Step 1: Calculate vanishing point
        vp = self.calculate_vanishing_point(lights_prev, lights_curr)
        
        if vp is None:
            # Parallel motion or tracking error
            return None
        
        # Step 2: Convert to 3D motion direction
        motion_direction = self.vanishing_point_to_3d_direction(vp)
        
        # Step 3: Classify motion type (TRANSLATION vs ROTATION)
        motion_type = self.classify_motion_type(lights_curr, motion_direction)
        
        # Step 4: Verify perpendicularity (translational motion)
        is_perpendicular = self.verify_perpendicularity(lights_curr, motion_direction)
        
        if not is_perpendicular and motion_type == "TRANSLATION":
            # Inconsistent: classified as translation but not perpendicular
            motion_type = "ROTATION"
        
        # Step 5: Estimate distance
        distance = self.estimate_distance_to_plane(lights_curr)
        
        # Step 6: Reconstruct pose
        rvec, tvec, R = self.reconstruct_pose(lights_curr, motion_direction, distance)
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'vanishing_point': vp,
            'distance': distance,
            'motion_type': motion_type,
            'is_valid': is_perpendicular,
            'frame': frame_idx
        }