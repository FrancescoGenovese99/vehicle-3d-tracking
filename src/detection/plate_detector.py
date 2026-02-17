"""
plate_detector.py

Automatic detection of a vehicle's rear license plate.

Detection pipeline
------------------
1. Gaussian blur       – reduces high-frequency sensor noise before edge detection.
2. Canny edge detection – extracts strong intensity gradients.
3. Morphological close  – bridges small gaps in the plate border contour.
4. Contour extraction   – finds closed regions in the edge map.
5. Geometric filtering  – keeps only quadrilaterals whose area, aspect ratio,
                           and solidity match a real license plate.
6. Corner ordering      – returns corners in a canonical TL→TR→BR→BL order
                           suitable for perspective transforms.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict


class PlateDetector:
    """
    Detects the rear license plate of a vehicle in a single BGR frame.

    The detector is intentionally stateless: each call to ``detect_plate_corners``
    processes the frame from scratch, making it safe to use across threads or
    to swap into any frame-level processing pipeline.

    Output convention
    -----------------
    All corner arrays follow the order:
        [top-left, top-right, bottom-right, bottom-left]
    This matches the destination layout expected by ``cv2.getPerspectiveTransform``.
    """

    def __init__(self, params: Dict):
        """
        Args:
            params: Full configuration dictionary (from detection_params.yaml).
                    Relevant sub-key: ``license_plate_detection``.
        """
        cfg = params.get('license_plate_detection', {})

        # --- Edge detection parameters ---
        edge_cfg = cfg.get('edge_detection', {})
        self.canny_low   = edge_cfg.get('canny_low',   50)
        self.canny_high  = edge_cfg.get('canny_high',  150)
        self.blur_kernel = edge_cfg.get('blur_kernel', 5)   # must be odd

        # --- Contour filtering thresholds ---
        contour_cfg = cfg.get('contour_filter', {})
        self.min_area          = contour_cfg.get('min_area',          1000)
        self.max_area          = contour_cfg.get('max_area',          50000)
        self.aspect_ratio_min  = contour_cfg.get('aspect_ratio_min',  2.5)
        self.aspect_ratio_max  = contour_cfg.get('aspect_ratio_max',  6.0)
        # A standard Italian plate is ~520 × 110 mm → aspect ratio ≈ 4.7
        self.min_solidity      = contour_cfg.get('min_solidity',      0.7)
        # Solidity = contour_area / convex_hull_area; plates are nearly convex

        # Epsilon controls how tightly the polygon follows the contour.
        # Smaller values → more vertices; larger → fewer, coarser approximation.
        self.epsilon_factor = cfg.get('epsilon_factor', 0.02)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_plate_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Locate the four corners of the license plate in a BGR frame.

        Args:
            frame: Input image in BGR format (as returned by OpenCV).

        Returns:
            Float32 array of shape (4, 2) with corners ordered
            [TL, TR, BR, BL], or None if no valid plate is found.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Morphological close: dilate then erode to bridge small edge gaps
        # that commonly appear on embossed or partially lit plate borders.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=1)
        edges  = cv2.erode(edges, kernel,  iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        plate_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Reject blobs that are too small (noise) or too large (vehicle body)
            if not (self.min_area <= area <= self.max_area):
                continue

            # Approximate the contour as a polygon; keep only quadrilaterals
            perimeter = cv2.arcLength(contour, True)
            approx    = cv2.approxPolyDP(contour, self.epsilon_factor * perimeter, True)
            if len(approx) != 4:
                continue

            # Aspect ratio check using the bounding rectangle
            x, y, w, h    = cv2.boundingRect(approx)
            aspect_ratio  = w / float(h) if h > 0 else 0
            if not (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max):
                continue

            # Solidity check: reject concave or irregular shapes
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity  = area / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                continue

            plate_candidates.append({
                'corners':      approx.reshape(4, 2),
                'area':         area,
                'aspect_ratio': aspect_ratio,
                'solidity':     solidity,
                'bbox':         (x, y, w, h),
            })

        if not plate_candidates:
            return None

        # Among all valid candidates, pick the largest (closest to camera)
        best = max(plate_candidates, key=lambda c: c['area'])

        return self._order_corners(best['corners'])

    # ------------------------------------------------------------------
    # Corner ordering
    # ------------------------------------------------------------------

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Reorder four unordered corner points into the canonical sequence
        [top-left, top-right, bottom-right, bottom-left].

        Strategy: split points into top half / bottom half relative to the
        centroid, then sort each half by x-coordinate. An angle-based fallback
        is used when the centroid split produces fewer than two points in
        either half (e.g. nearly horizontal plates).

        Args:
            corners: Unordered (4, 2) array of corner coordinates.

        Returns:
            Ordered (4, 2) float32 array.
        """
        center = corners.mean(axis=0)

        top_mask    = corners[:, 1] < center[1]
        top_points  = corners[top_mask]
        bot_points  = corners[~top_mask]

        if len(top_points) >= 2:
            top_left  = top_points[np.argmin(top_points[:, 0])]
            top_right = top_points[np.argmax(top_points[:, 0])]
        else:
            # Angle-based fallback: sort CCW from the leftmost point
            angles        = np.arctan2(corners[:, 1] - center[1],
                                       corners[:, 0] - center[0])
            sorted_idx    = np.argsort(angles)
            top_left      = corners[sorted_idx[0]]
            top_right     = corners[sorted_idx[1]]

        if len(bot_points) >= 2:
            bottom_left  = bot_points[np.argmin(bot_points[:, 0])]
            bottom_right = bot_points[np.argmax(bot_points[:, 0])]
        else:
            angles        = np.arctan2(corners[:, 1] - center[1],
                                       corners[:, 0] - center[0])
            sorted_idx    = np.argsort(angles)
            bottom_right  = corners[sorted_idx[2]]
            bottom_left   = corners[sorted_idx[3]]

        return np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    def detect_plate_region(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect the plate and return both its corner coordinates and a
        perspective-corrected crop of the plate region.

        The crop is obtained via ``cv2.warpPerspective``, so it is always
        a frontal, rectangular view regardless of camera angle.

        Args:
            frame: Input BGR frame.

        Returns:
            (corners, plate_crop) where corners is (4, 2) float32 and
            plate_crop is a BGR image of the rectified plate, or None if
            detection failed.
        """
        corners = self.detect_plate_corners(frame)
        if corners is None:
            return None

        # Compute output dimensions from the detected corner positions
        width  = int(np.linalg.norm(corners[1] - corners[0]))
        height = int(np.linalg.norm(corners[3] - corners[0]))

        dst_corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=np.float32
        )

        M          = cv2.getPerspectiveTransform(corners, dst_corners)
        plate_crop = cv2.warpPerspective(frame, M, (width, height))

        return corners, plate_crop

    def refine_corners_subpixel(
        self, frame: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """
        Refine coarse corner positions to sub-pixel accuracy using the
        iterative ``cv2.cornerSubPix`` algorithm.

        This step is optional but improves the geometric precision of the
        perspective transform, which matters when the plate crop is used
        for OCR or precise measurement.

        Args:
            frame:   Input BGR frame (used to extract the grayscale channel).
            corners: Initial corner estimates from ``detect_plate_corners``.

        Returns:
            Refined (4, 2) float32 corner array.
        """
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        return cv2.cornerSubPix(
            gray,
            corners.astype(np.float32),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def visualize_detection(
        self,
        frame: np.ndarray,
        corners: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a debug overlay on top of the input frame.

        - If corners are provided: draws the plate polygon, corner dots, and
          TL/TR/BR/BL labels.
        - If corners are None: blends the Canny edge map over the frame and
          prints a "No plate detected" banner, helping to diagnose why
          detection failed.

        Args:
            frame:   Input BGR frame.
            corners: Detected corners (4, 2), or None.

        Returns:
            BGR image with debug overlay (same resolution as ``frame``).
        """
        vis = frame.copy()

        if corners is None:
            # Overlay the edge map so the caller can see what the detector sees
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
            edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)
            vis     = cv2.addWeighted(vis, 0.7,
                                      cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.putText(vis, "No plate detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # Draw plate boundary polygon
            cv2.polylines(vis, [corners.astype(int)], isClosed=True,
                          color=(0, 255, 0), thickness=3)

            # Mark each corner with a dot and its canonical label
            labels = ['TL', 'TR', 'BR', 'BL']
            for label, corner in zip(labels, corners):
                pt = tuple(corner.astype(int))
                cv2.circle(vis, pt, 8, (0, 255, 0), thickness=-1)
                cv2.putText(vis, label, (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(vis, "Plate detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        return vis


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def detect_plate_corners(
    frame: np.ndarray, params: Dict
) -> Optional[np.ndarray]:
    """
    Stateless convenience wrapper around ``PlateDetector.detect_plate_corners``.

    Instantiates a fresh ``PlateDetector`` on every call, so it is suitable
    for one-off detections or scripts that do not need a persistent object.

    Args:
        frame:  Input BGR frame.
        params: Configuration dictionary (same format as ``PlateDetector``).

    Returns:
        (4, 2) float32 corner array ordered [TL, TR, BR, BL], or None.
    """
    return PlateDetector(params).detect_plate_corners(frame)