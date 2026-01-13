"""
License Plate Detector - Rilevamento automatico targa posteriore.

Rileva i 4 angoli della targa usando:
1. Edge detection (Canny)
2. Contour detection
3. Filtri geometrici (aspect ratio, area)
4. Approximazione poligonale a 4 punti
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict


class PlateDetector:
    """
    Rileva automaticamente la targa posteriore del veicolo.
    Output: 4 angoli ordinati (top-left, top-right, bottom-right, bottom-left)
    """
    
    def __init__(self, params: Dict):
        """
        Initialize detector.
        
        Args:
            params: Detection parameters from detection_params.yaml
        """
        self.params = params.get('license_plate_detection', {})
        
        # Edge detection
        edge_cfg = self.params.get('edge_detection', {})
        self.canny_low = edge_cfg.get('canny_low', 50)
        self.canny_high = edge_cfg.get('canny_high', 150)
        self.blur_kernel = edge_cfg.get('blur_kernel', 5)
        
        # Contour filtering
        contour_cfg = self.params.get('contour_filter', {})
        self.min_area = contour_cfg.get('min_area', 1000)
        self.max_area = contour_cfg.get('max_area', 50000)
        self.aspect_ratio_min = contour_cfg.get('aspect_ratio_min', 2.5)
        self.aspect_ratio_max = contour_cfg.get('aspect_ratio_max', 6.0)
        self.min_solidity = contour_cfg.get('min_solidity', 0.7)
        
        # Polygon approximation
        self.epsilon_factor = self.params.get('epsilon_factor', 0.02)
    
    def detect_plate_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect 4 corners of license plate.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Array (4, 2) with corners [top-left, top-right, bottom-right, bottom-left]
            or None if detection failed
        """
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter and find best plate candidate
        plate_candidates = []
        
        for contour in contours:
            # Area filter
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Must be 4-sided (quadrilateral)
            if len(approx) != 4:
                continue
            
            # Bounding rectangle for aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Aspect ratio filter (Italian plate ~520x110mm = 4.7:1)
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                continue
            
            # Solidity filter (filled ratio)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < self.min_solidity:
                continue
            
            # Valid candidate
            plate_candidates.append({
                'corners': approx.reshape(4, 2),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'bbox': (x, y, w, h)
            })
        
        if not plate_candidates:
            return None
        
        # Select best candidate (largest area with good aspect ratio)
        best_candidate = max(plate_candidates, key=lambda c: c['area'])
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(best_candidate['corners'])
        
        return corners
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent way: TL, TR, BR, BL.
        
        Args:
            corners: Unordered corners (4, 2)
            
        Returns:
            Ordered corners (4, 2)
        """
        # Calculate center
        center = corners.mean(axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], 
                           corners[:, 0] - center[0])
        
        # Find top-left (smallest angle in top half)
        top_points = corners[corners[:, 1] < center[1]]
        if len(top_points) >= 2:
            top_left = top_points[np.argmin(top_points[:, 0])]
            top_right = top_points[np.argmax(top_points[:, 0])]
        else:
            # Fallback: use angles
            sorted_indices = np.argsort(angles)
            top_left = corners[sorted_indices[0]]
            top_right = corners[sorted_indices[1]]
        
        # Bottom points
        bottom_points = corners[corners[:, 1] >= center[1]]
        if len(bottom_points) >= 2:
            bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
            bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
        else:
            sorted_indices = np.argsort(angles)
            bottom_right = corners[sorted_indices[2]]
            bottom_left = corners[sorted_indices[3]]
        
        # Order: TL, TR, BR, BL
        ordered = np.array([top_left, top_right, bottom_right, bottom_left], 
                          dtype=np.float32)
        
        return ordered
    
    def visualize_detection(self, frame: np.ndarray, 
                           corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize detection for debugging.
        
        Args:
            frame: Input frame
            corners: Detected corners (4, 2) or None
            
        Returns:
            Frame with visualization
        """
        vis = frame.copy()
        
        if corners is None:
            # Show edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # Convert edges to BGR for overlay
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(vis, 0.7, edges_bgr, 0.3, 0)
            
            cv2.putText(vis, "No plate detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # Draw corners
            for i, corner in enumerate(corners):
                pt = tuple(corner.astype(int))
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
                
                # Labels
                labels = ['TL', 'TR', 'BR', 'BL']
                cv2.putText(vis, labels[i], (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw plate boundary
            cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 3)
            
            cv2.putText(vis, "Plate detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return vis
    
    def detect_plate_region(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect plate and return corners + cropped region.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple (corners, plate_crop) or None
        """
        corners = self.detect_plate_corners(frame)
        
        if corners is None:
            return None
        
        # Warp perspective to get plate crop
        width = int(np.linalg.norm(corners[1] - corners[0]))
        height = int(np.linalg.norm(corners[3] - corners[0]))
        
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        plate_crop = cv2.warpPerspective(frame, M, (width, height))
        
        return corners, plate_crop
    
    def refine_corners_subpixel(self, frame: np.ndarray, 
                                corners: np.ndarray) -> np.ndarray:
        """
        Refine corner positions to sub-pixel accuracy.
        
        Args:
            frame: Input frame (BGR)
            corners: Initial corner positions (4, 2)
            
        Returns:
            Refined corners (4, 2)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Corner refinement parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        
        corners_refined = cv2.cornerSubPix(
            gray,
            corners.astype(np.float32),
            win_size,
            zero_zone,
            criteria
        )
        
        return corners_refined


# Standalone function for compatibility
def detect_plate_corners(frame: np.ndarray, params: Dict) -> Optional[np.ndarray]:
    """
    Convenience function for plate detection.
    
    Args:
        frame: Input frame
        params: Detection parameters
        
    Returns:
        4 corners or None
    """
    detector = PlateDetector(params)
    return detector.detect_plate_corners(frame)