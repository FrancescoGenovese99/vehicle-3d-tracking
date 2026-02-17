"""
Light Detector

Detects rear tail lights in a video frame using HSV colour filtering
and contour-based blob detection.

Outputs a list of LightCandidate objects, each describing one detected
light blob.  Downstream, CandidateSelector picks the best left/right pair.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LightCandidate:
    """
    A single detected light blob.

    Attributes:
        center      : (x, y) centroid in pixel coordinates
        contour     : OpenCV contour array
        area        : blob area in pixels²
        circularity : compactness ratio in [0, 1]  (circle = 1)
        bbox        : axis-aligned bounding box (x, y, w, h)
    """
    center:      Tuple[int, int]
    contour:     np.ndarray
    area:        float
    circularity: float
    bbox:        Tuple[int, int, int, int]


class LightDetector:
    """
    Detects rear (and optionally front) vehicle lights using HSV masking.

    Red lights wrap around the HSV hue axis (0-10° and 170-180°), so two
    separate masks are combined.  White lights (reverse / plate illumination)
    use a single low-saturation, high-value range.

    All thresholds are loaded from detection_params.yaml so they can be
    tuned without modifying this file.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config : dict loaded from detection_params.yaml
        """
        self.config = config

        # --- HSV colour ranges ---
        hsv_cfg  = config.get('hsv_ranges', {})
        red_cfg  = hsv_cfg.get('red',   {})
        white_cfg = hsv_cfg.get('white', {})

        # Red is split across the hue wrap-around (0° and 180° are both red)
        self.red_lower1 = np.array(red_cfg.get('lower1', [  0, 100, 100]))
        self.red_upper1 = np.array(red_cfg.get('upper1', [ 10, 255, 255]))
        self.red_lower2 = np.array(red_cfg.get('lower2', [170, 100, 100]))
        self.red_upper2 = np.array(red_cfg.get('upper2', [180, 255, 255]))

        self.white_lower = np.array(white_cfg.get('lower', [  0,  0, 200]))
        self.white_upper = np.array(white_cfg.get('upper', [180, 30, 255]))

        # --- Blob filtering ---
        blob_cfg             = config.get('blob_detection', {})
        self.min_area        = blob_cfg.get('min_area',        50)
        self.max_area        = blob_cfg.get('max_area',      5000)
        self.min_circularity = blob_cfg.get('min_circularity', 0.4)

        # --- Morphology kernel ---
        morph_cfg             = config.get('morphology', {})
        kernel_size           = morph_cfg.get('kernel_size', [5, 5])
        self.kernel           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          tuple(kernel_size))
        self.open_iterations  = morph_cfg.get('open_iterations',  1)
        self.close_iterations = morph_cfg.get('close_iterations', 1)

    # ------------------------------------------------------------------
    # Mask creation
    # ------------------------------------------------------------------

    def detect_red_lights(self, frame: np.ndarray) -> np.ndarray:
        """
        Build a binary mask for red pixels (rear tail lights).

        Combines two HSV ranges to cover the hue wrap-around at 0°/180°.

        Args:
            frame : BGR image

        Returns:
            Binary mask (same H×W as frame)
        """
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        return cv2.bitwise_or(mask1, mask2)

    def detect_white_lights(self, frame: np.ndarray) -> np.ndarray:
        """
        Build a binary mask for white pixels (plate lamp, reverse light).

        Args:
            frame : BGR image

        Returns:
            Binary mask (same H×W as frame)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.white_lower, self.white_upper)

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean a binary mask with morphological opening then closing.

        Opening  : removes small isolated noise blobs.
        Closing  : fills small holes inside detected regions.

        Args:
            mask : binary mask

        Returns:
            Cleaned binary mask
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel,
                                iterations=self.open_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel,
                                iterations=self.close_iterations)
        return mask

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def find_light_candidates(self, mask: np.ndarray) -> List[LightCandidate]:
        """
        Extract LightCandidate objects from a binary mask.

        Filters by area and circularity; tail lights tend to be
        roughly elliptical, so very elongated or jagged blobs are rejected.

        Args:
            mask : cleaned binary mask

        Returns:
            List of LightCandidate (may be empty)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            candidates.append(LightCandidate(
                center=(cx, cy),
                contour=cnt,
                area=area,
                circularity=circularity,
                bbox=cv2.boundingRect(cnt)
            ))

        return candidates

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def detect_tail_lights(self, frame: np.ndarray,
                           combine_masks: bool = True
                           ) -> Tuple[List[LightCandidate], np.ndarray]:
        """
        Detect tail light candidates in a single frame.

        Args:
            frame         : BGR image
            combine_masks : if True, merges red and white masks before
                            blob detection (catches plate lamp overlap)

        Returns:
            (candidates, combined_mask)
        """
        red_mask   = self.detect_red_lights(frame)
        white_mask = self.detect_white_lights(frame)

        combined = cv2.bitwise_or(red_mask, white_mask) if combine_masks else red_mask
        combined = self.apply_morphology(combined)

        return self.find_light_candidates(combined), combined

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def visualize_detection(self, frame: np.ndarray,
                            candidates: List[LightCandidate],
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detected candidates on a copy of the frame for debugging.

        Args:
            frame      : original BGR image
            candidates : list of detected candidates
            mask       : optional binary mask to append side-by-side

        Returns:
            Annotated frame (mask appended on the right if provided)
        """
        vis = frame.copy()

        for c in candidates:
            cv2.drawContours(vis, [c.contour], -1, (0, 255, 0), 2)
            cv2.circle(vis, c.center, 5, (0, 0, 255), -1)

            x, y, w, h = c.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)

            label = f"A:{c.area:.0f} C:{c.circularity:.2f}"
            cv2.putText(vis, label, (c.center[0] + 10, c.center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if mask is not None:
            vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        return vis