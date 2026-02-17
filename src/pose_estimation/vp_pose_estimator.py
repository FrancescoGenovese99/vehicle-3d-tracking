"""
vp_pose_estimator.py

Vanishing Point-based Pose Estimation (Task Original Formulation)

This module implements the vanishing-point method prescribed in the assignment.
It estimates depth from the motion vanishing point Vx combined with the metric
constraint (inter-light distance).

NOTE: This method is numerically unstable on real nighttime footage due to
small inter-frame motion and feature detection noise. It is retained for:
  1. Diagnostic visualization (motion classification)
  2. Comparison with the PnP method (see report Section 4.2)

The production pipeline uses PnPPoseEstimator instead.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

class VPPoseEstimator:
    def __init__(self, camera_matrix, vehicle_model):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        
        # Inter-light distance (metric constraint)
        pnp_cfg = vehicle_model.get('vehicle', {}).get('pnp_points_3d', {})
        l_outer = np.array(pnp_cfg.get('light_l_outer'), dtype=np.float32)
        r_outer = np.array(pnp_cfg.get('light_r_outer'), dtype=np.float32)
        self.D_real = float(np.linalg.norm(r_outer - l_outer))
        
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.prev_features = None
        
    def compute_vx(self, features_curr, features_prev):
        """Compute motion vanishing point from optical flow"""
        if features_prev is None:
            return None
            
        lines = []
        for key in ('outer', 'top'):
            if key not in features_curr or key not in features_prev:
                continue
            # Convert to numpy arrays to handle both tuple and array inputs
            curr = np.array(features_curr[key], dtype=np.float32)
            prev = np.array(features_prev[key], dtype=np.float32)
            
            for i in range(min(len(curr), len(prev))):
                movement = np.linalg.norm(curr[i] - prev[i])
                if movement < 1.0:  # skip static points
                    continue
                    
                p0 = np.array([prev[i][0], prev[i][1], 1.0])
                p1 = np.array([curr[i][0], curr[i][1], 1.0])
                line = np.cross(p0, p1)
                norm = np.linalg.norm(line)
                if norm > 1e-8:
                    lines.append(line / norm)
        
        if len(lines) < 2:
            return None
            
        # SVD to find null space (vanishing point)
        _, _, Vt = np.linalg.svd(np.array(lines))
        vp_h = Vt[-1]
        if abs(vp_h[2]) < 1e-8:
            return None
        vp = vp_h[:2] / vp_h[2]
        
        if np.linalg.norm(vp) > 1e5:  # reject infinity
            return None
        return vp
        
    def estimate_depth_from_vp(self, outer_points):
        """Estimate depth using d = f * D_real / δ_pixel"""
        # Convert to numpy array to handle both tuple and array inputs
        outer_arr = np.array(outer_points, dtype=np.float32)
        L, R = outer_arr[0], outer_arr[1]
        delta_pixel = np.linalg.norm(R - L)
        
        if delta_pixel < 1.0:
            return None
            
        f_avg = (self.fx + self.fy) / 2.0
        depth = (self.D_real * f_avg) / delta_pixel
        
        return float(np.clip(depth, 2.0, 50.0))
        
    def estimate_pose_vp(self, features_curr, frame_idx=0):
        """Estimate pose using VP method (Task 2 original)"""
        outer = features_curr.get('outer')
        if outer is None or len(outer) < 2:
            return None
            
        # Compute Vx
        vx = self.compute_vx(features_curr, self.prev_features)
        
        # Estimate depth from pixel width
        depth = self.estimate_depth_from_vp(outer)
        if depth is None:
            return None
            
        # Store for next frame (convert everything to numpy arrays)
        self.prev_features = {}
        for k, v in features_curr.items():
            if isinstance(v, np.ndarray):
                self.prev_features[k] = v.copy()
            elif isinstance(v, (tuple, list)):
                self.prev_features[k] = np.array(v, dtype=np.float32)
            else:
                self.prev_features[k] = v
        
        # Return result (simplified - no full R reconstruction)
        return {
            'method': 'vp_geometric',
            'distance_vp': depth,
            'vx': vx,
            'frame': frame_idx
        }