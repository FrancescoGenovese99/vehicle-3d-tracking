"""
advanced_detector.py

Multi-feature vehicle keypoint detector with adaptive temporal stability.

Detection pipeline
------------------
1. Tail-light detection  – red-channel HSV thresholding followed by contour
                           filtering on area, aspect ratio, and vertical extent.
2. Feature extraction    – three sub-points are extracted per light blob:
                             'outer'  : lateral-most edge (via Canny scan)
                             'top'    : highest contour pixel
                             'bottom' : 90th-percentile Y point (see note below)
3. Plate detection       – a brightness-based blob search within a geometric
                           ROI derived from the tail-light positions, with
                           RANSAC line fitting for robust corner estimation.
4. Temporal stability    – an adaptive jump threshold (base + velocity-scaleda
                           margin) rejects spurious detections between frames.
                           Velocity is estimated from a short position history
                           and used both for threshold scaling and for
                           predicting the plate position when detection fails.

Bottom-point note
-----------------
argmax on Y would select the lowest pixel, which in practice often falls on
road-surface reflections or the bumper rather than the light housing itself.
Using the 90th-percentile Y boundary and averaging the points above it gives
a stable estimate anchored to the bottom edge of the actual light body.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy import stats
from collections import deque


@dataclass
class VehicleKeypoints:
    """
    Container for all keypoints detected on a single frame.

    Attributes:
        tail_lights_features: Dict with keys 'outer', 'top', 'bottom', each a
                              (2, 2) float32 array [[left_x, left_y],
                              [right_x, right_y]].
        plate_corners:        Dict with keys 'TL', 'TR', 'BL', 'BR' mapping to
                              (x, y) integer pixel coordinates, or None.
        plate_bottom:         (2, 2) float32 array [[BL_x, BL_y], [BR_x, BR_y]],
                              or None.
        plate_center:         (x, y) centroid of the detected plate region.
        confidence:           Detection confidence in [0, 1].
        templates:            Optional dict of grayscale template patches keyed
                              by feature name, used for template-matching tracking.
    """
    tail_lights_features: Dict[str, np.ndarray]
    plate_corners:        Optional[Dict[str, Tuple[int, int]]]
    plate_bottom:         Optional[np.ndarray]
    plate_center:         Tuple[int, int]
    confidence:           float
    templates:            Optional[Dict[str, List[np.ndarray]]] = None


class AdvancedDetector:
    """
    Multi-feature detector for tail lights and license plate.

    The detector is designed to handle the full distance range of a following
    vehicle: from very close (large blobs, wide plate) to far (small blobs,
    narrow plate). Adaptive thresholds and velocity prediction keep the
    output stable across frames even when individual detections are noisy.
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config: Optional override dict. Recognised keys:
                      'v_lower'      – minimum V channel for red-light mask
                      'v_plate_low'  – minimum V for plate blob
                      'v_plate_high' – maximum V for plate blob
        """
        # --- HSV thresholds for tail-light (red) detection ---
        # Red wraps around the hue circle, so two ranges are needed.
        self.red_h_lower1 = 0
        self.red_h_upper1 = 18
        self.red_h_lower2 = 170
        self.red_h_upper2 = 180

        self.red_s_lower = 135
        self.red_s_upper = 255

        self.red_v_lower = 190   # high brightness to select lit lights only
        self.red_v_upper = 255

        # --- Contour geometry filters ---
        self.min_contour_area_ratio = 0.00015   # fraction of frame area
        self.max_contour_area_ratio = 0.01
        self.min_vertical_ratio     = 1.2        # h/w: tail lights are taller than wide
        self.max_y_ratio            = 0.80       # ignore blobs in the bottom 20 % of frame

        # --- Sub-feature extraction parameters ---
        self.mid_height_min = 0.30   # fraction of blob height for Canny scan window
        self.mid_height_max = 0.70

        # --- Canny parameters (used inside the light ROI) ---
        self.canny_low  = 50
        self.canny_high = 150

        # Template patch half-size for tracking initialisation
        self.template_size = 35

        # --- Plate detection brightness range ---
        self.v_plate_low  = 150
        self.v_plate_high = 240

        # --- Temporal stability state ---
        self.prev_plate_bottom  = None
        self.prev_plate_corners = None

        # Base jump threshold; increased during the warmup phase so that the
        # first few frames do not reject legitimate detections.
        self.base_max_jump_pixels = 80
        self.max_jump_pixels      = self.base_max_jump_pixels
        self.plate_detection_count = 0
        self.warmup_frames         = 10

        # Rolling plate-centre history for velocity estimation
        self.plate_center_history = deque(maxlen=5)
        self.plate_velocity       = np.array([0.0, 0.0])

        # Confidence decays when detections are rejected and recovers otherwise
        self.detection_confidence = 1.0
        self.min_confidence       = 0.3

        # Apply any caller-provided overrides
        if config:
            self.red_v_lower  = config.get('v_lower',      self.red_v_lower)
            self.v_plate_low  = config.get('v_plate_low',  self.v_plate_low)
            self.v_plate_high = config.get('v_plate_high', self.v_plate_high)

        print("[AdvancedDetector] Initialized (multi-feature, adaptive threshold)")
        print(f"  Tail lights: HSV V >= {self.red_v_lower}")
        print(f"  Plate      : blob + RANSAC, V in [{self.v_plate_low}, {self.v_plate_high}]")
        print(f"  Jump thresh: adaptive (base = {self.base_max_jump_pixels} px)")

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _create_red_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Build a binary mask selecting bright-red pixels (tail lights).

        Only the V (brightness) channel is thresholded here because the
        illuminated portion of a tail light is nearly saturated; the full
        HSV range check is applied later during contour filtering.

        Args:
            frame: BGR input image.

        Returns:
            Binary mask (same H × W as frame).
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv[:, :, 2], self.red_v_lower, 255)
        return mask

    # ------------------------------------------------------------------
    # Per-light feature extraction
    # ------------------------------------------------------------------

    def _extract_outer_point_with_canny(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        is_left: bool,
    ) -> Tuple[int, int]:
        """
        Find the lateral-most edge point of a light blob in the mid-height band.

        A small padded ROI is cropped around the bounding box and Canny edges
        are computed within it. Scanning each row of the mid-height band and
        taking the left-most (or right-most) edge pixel gives a sub-blob
        precision outer point that is less affected by blob shape irregularities
        than simply using the bounding-box corner.

        If no Canny edges are found, the method falls back to contour extrema.

        Args:
            frame:    Full BGR frame.
            contour:  Raw contour of the light blob.
            bbox:     (x, y, w, h) bounding rectangle of the contour.
            is_left:  True for the left light (seek left-most edge),
                      False for the right light (seek right-most edge).

        Returns:
            (x, y) pixel coordinate of the outer point in frame coordinates.
        """
        x, y, w, h = bbox
        pad = 5
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return (x if is_left else x + w, y + h // 2)

        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                          self.canny_low, self.canny_high)

        roi_h     = y2 - y1
        mid_y_min = int(roi_h * self.mid_height_min)
        mid_y_max = int(roi_h * self.mid_height_max)

        candidates = []
        for py in range(mid_y_min, mid_y_max):
            if py >= edges.shape[0]:
                continue
            edge_pixels = np.where(edges[py, :] > 0)[0]
            if len(edge_pixels) == 0:
                continue
            px = edge_pixels[0] if is_left else edge_pixels[-1]
            candidates.append((px, py))

        if not candidates:
            # Fallback: use contour points in the mid-height band
            mid_pts = contour[
                (contour[:, 0, 1] >= y + int(h * self.mid_height_min)) &
                (contour[:, 0, 1] <= y + int(h * self.mid_height_max))
            ]
            if len(mid_pts) > 0:
                idx = (mid_pts[:, 0, 0].argmin() if is_left
                       else mid_pts[:, 0, 0].argmax())
                return tuple(mid_pts[idx, 0])
            return (x if is_left else x + w, y + h // 2)

        candidates = np.array(candidates)
        return (
            int(np.median(candidates[:, 0])) + x1,
            int(np.median(candidates[:, 1])) + y1,
        )

    def _get_robust_bottom_point(self, contour: np.ndarray) -> Tuple[int, int]:
        """
        Estimate the bottom edge of a light blob, excluding surface reflections.

        The lowest-Y pixel (argmax on Y) tends to fall on road or bumper
        reflections. This method instead takes the band between the 90th and
        100th percentile of Y values and averages it, anchoring the result to
        the actual bottom edge of the light housing.

        Args:
            contour: Raw contour of the light blob.

        Returns:
            (x, y) averaged bottom point in frame coordinates.
        """
        points       = contour[:, 0, :]
        sorted_by_y  = points[points[:, 1].argsort()]

        # Retain only the top-10 % by Y (the true bottom band of the light)
        idx_90th         = int(len(sorted_by_y) * 0.90)
        relevant_points  = sorted_by_y[idx_90th:]

        if len(relevant_points) == 0:
            relevant_points = sorted_by_y[-5:]  # safety fallback

        return (int(np.mean(relevant_points[:, 0])),
                int(np.mean(relevant_points[:, 1])))

    def _extract_feature_points(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        is_left: bool,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract the three canonical feature points for one tail-light blob.

        Args:
            frame:    Full BGR frame.
            contour:  Raw contour of the light blob.
            is_left:  Indicates which light side (affects outer-point search direction).

        Returns:
            Dict with keys 'top', 'bottom', 'outer', each mapped to an (x, y) tuple.
        """
        x, y, w, h = cv2.boundingRect(contour)

        features = {
            'top':    tuple(contour[contour[:, 0, 1].argmin(), 0]),
            'bottom': self._get_robust_bottom_point(contour),
            'outer':  self._extract_outer_point_with_canny(
                          frame, contour, (x, y, w, h), is_left),
        }
        return features

    # ------------------------------------------------------------------
    # Tail-light detection
    # ------------------------------------------------------------------

    def detect_tail_lights_multifeature(
        self, frame: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Detect the tail-light pair and return their multi-feature keypoints.

        Pair selection maximises horizontal separation while penalising
        vertical misalignment and large area asymmetry.

        Args:
            frame: BGR input frame.

        Returns:
            Dict with keys 'top', 'outer', 'bottom', each a (2, 2) float32
            array [[left_x, left_y], [right_x, right_y]], or None if no
            valid pair is found.
        """
        height, width = frame.shape[:2]
        frame_area    = width * height

        min_area = self.min_contour_area_ratio * frame_area
        max_area = self.max_contour_area_ratio * frame_area
        max_y    = int(height * self.max_y_ratio)

        # Build and clean the red-light mask
        mask   = self._create_red_mask(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area, position, and aspect ratio
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue

            x, y, w, h = cv2.boundingRect(c)
            if y > max_y or w == 0 or h == 0:
                continue
            if w / h > 1.0 or h / w < self.min_vertical_ratio:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            candidates.append((c, cx, cy, area))

        if len(candidates) < 2:
            return None

        # Score all pairs and keep the best
        best_pair  = None
        best_score = 0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, x1, y1, a1 = candidates[i]
                c2, x2, y2, a2 = candidates[j]

                dx         = abs(x2 - x1)
                dy         = abs(y2 - y1)
                area_ratio = min(a1, a2) / max(a1, a2)

                # Minimum separation, vertical alignment, and size symmetry checks
                if dx < 80 or dy > 50 or area_ratio < 0.3:
                    continue

                score = dx * 2.0 - dy * 1.0 + area_ratio * 50
                if score > best_score:
                    best_score = score
                    best_pair  = (c1, c2) if x1 < x2 else (c2, c1)

        if best_pair is None:
            return None

        left_contour, right_contour = best_pair
        left_features  = self._extract_feature_points(frame, left_contour,  is_left=True)
        right_features = self._extract_feature_points(frame, right_contour, is_left=False)

        features_dict = {}
        for name in ('top', 'outer', 'bottom'):
            features_dict[name] = np.array(
                [left_features[name], right_features[name]], dtype=np.float32
            )

        return features_dict

    def _extract_keypoint_template(
        self, frame: np.ndarray, center: Tuple[int, int]
    ) -> np.ndarray:
        """
        Crop a square grayscale patch centred on a keypoint.

        The patch is used as a template for subsequent frame-to-frame tracking.
        Patches that fall partially outside the frame are resized to the nominal
        template size to keep the output shape consistent.

        Args:
            frame:  Full BGR frame.
            center: (x, y) pixel coordinate of the keypoint.

        Returns:
            Grayscale patch of shape (template_size, template_size).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = center
        half   = self.template_size // 2

        x1 = max(0, cx - half);  x2 = min(gray.shape[1], cx + half + 1)
        y1 = max(0, cy - half);  y2 = min(gray.shape[0], cy + half + 1)

        patch = gray[y1:y2, x1:x2].copy()
        if patch.shape[0] < self.template_size or patch.shape[1] < self.template_size:
            patch = cv2.resize(patch, (self.template_size, self.template_size))
        return patch

    def detect_tail_lights_with_templates(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, List[np.ndarray]]]]:
        """
        Detect tail lights and extract template patches for each keypoint.

        Args:
            frame: BGR input frame.

        Returns:
            (features_dict, templates_dict) or (None, None) if detection fails.
            templates_dict mirrors features_dict's structure but stores grayscale
            patch arrays instead of coordinate arrays.
        """
        features_dict = self.detect_tail_lights_multifeature(frame)
        if features_dict is None:
            return None, None

        templates_dict = {}
        for name, points in features_dict.items():
            templates_dict[name] = [
                self._extract_keypoint_template(frame, tuple(map(int, points[i])))
                for i in range(2)
            ]

        return features_dict, templates_dict

    # ------------------------------------------------------------------
    # Plate detection
    # ------------------------------------------------------------------

    def _fit_line_ransac(
        self, points: np.ndarray, axis: str = 'horizontal'
    ) -> Optional[Tuple[float, float]]:
        """
        Fit a line to a set of 2-D points using RANSAC.

        RANSAC is used instead of ordinary least squares because plate blob
        boundaries frequently include outlier pixels from shadows, text, or
        reflections.

        Args:
            points: (N, 2) array of (x, y) coordinates.
            axis:   'horizontal' fits  y = m*x + q  (for top/bottom edges).
                    'vertical'   fits  x = m*y + q  (for left/right edges).

        Returns:
            (slope, intercept) of the fitted line, or None if fitting failed.
        """
        if len(points) < 2:
            return None

        points = np.array(points)

        if axis == 'horizontal':
            X, y_vals = points[:, 0].reshape(-1, 1), points[:, 1]
            if np.std(X) < 1e-6:
                return None
        else:
            X, y_vals = points[:, 1].reshape(-1, 1), points[:, 0]
            if np.std(X) < 1e-6:
                return None

        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor(residual_threshold=3.0, max_trials=100,
                                     min_samples=2, random_state=42)
            ransac.fit(X, y_vals)
            return float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_)
        except Exception:
            # Fall back to ordinary least squares if sklearn is unavailable
            if axis == 'horizontal':
                slope, intercept, *_ = stats.linregress(points[:, 0], points[:, 1])
            else:
                slope, intercept, *_ = stats.linregress(points[:, 1], points[:, 0])
            return float(slope), float(intercept)

    def _process_plate_in_roi(
        self,
        frame: np.ndarray,
        roi: np.ndarray,
        roi_offset: Tuple[int, int],
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Detect plate corners within a pre-computed ROI.

        The method segments the plate blob by brightness thresholding, then
        splits its contour points into four edge bands and fits a RANSAC line
        to each. Corner coordinates are recovered from the line intersections.
        If any intersection cannot be computed, a minimum-area rectangle
        (``cv2.minAreaRect``) is used as a fallback.

        Args:
            frame:      Full BGR frame (unused directly; kept for API consistency).
            roi:        BGR sub-image of the region where the plate is expected.
            roi_offset: (x, y) top-left corner of the ROI in frame coordinates,
                        used to convert ROI-local coordinates back to frame coords.

        Returns:
            Dict {'TL', 'TR', 'BL', 'BR'} with (x, y) frame-coordinate corners,
            or None if the plate cannot be localised.
        """
        if roi.size == 0 or roi.shape[1] < 50:
            return None

        try:
            hsv_roi    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_plate = cv2.inRange(hsv_roi[:, :, 2], self.v_plate_low, self.v_plate_high)

            # Close small holes in the plate blob
            kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)

            contours_plate, _ = cv2.findContours(mask_plate, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            if not contours_plate:
                return None

            largest = max(contours_plate, key=cv2.contourArea)
            points  = largest.reshape(-1, 2)

            if len(points) < 10:
                return None

            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            h_plate = y_max - y_min
            w_plate = x_max - x_min

            if w_plate < 15 or h_plate < 5:
                return None

            # Define edge bands: inner 70 % of width/height to avoid corner ambiguity
            top_band    = y_min + 0.25 * h_plate
            bottom_band = y_max - 0.25 * h_plate
            left_band   = x_min + 0.15 * w_plate
            right_band  = x_max - 0.15 * w_plate

            top_pts, bottom_pts, left_pts, right_pts = [], [], [], []
            for x, y in points:
                if y <= top_band    and left_band <= x <= right_band: top_pts.append([x, y])
                if y >= bottom_band and left_band <= x <= right_band: bottom_pts.append([x, y])
                if x <= left_band   and top_band  <= y <= bottom_band: left_pts.append([x, y])
                if x >= right_band  and top_band  <= y <= bottom_band: right_pts.append([x, y])

            top_line    = self._fit_line_ransac(np.array(top_pts),    'horizontal') if len(top_pts)    >= 2 else None
            bottom_line = self._fit_line_ransac(np.array(bottom_pts), 'horizontal') if len(bottom_pts) >= 2 else None
            left_line   = self._fit_line_ransac(np.array(left_pts),   'vertical')   if len(left_pts)   >= 2 else None
            right_line  = self._fit_line_ransac(np.array(right_pts),  'vertical')   if len(right_pts)  >= 2 else None

            def intersect_h_v(h_line, v_line):
                """Intersect  y = m_h*x + q_h  with  x = m_v*y + q_v."""
                if h_line is None or v_line is None:
                    return None
                m_h, q_h = h_line
                m_v, q_v = v_line
                denom = 1 - m_h * m_v
                if abs(denom) < 1e-6:
                    return None
                y = (m_h * q_v + q_h) / denom
                x = m_v * y + q_v
                return (int(x), int(y))

            TL = intersect_h_v(top_line,    left_line)
            TR = intersect_h_v(top_line,    right_line)
            BL = intersect_h_v(bottom_line, left_line)
            BR = intersect_h_v(bottom_line, right_line)

            if not all([TL, TR, BL, BR]):
                # Fallback: minimum-area enclosing rectangle
                rect = cv2.minAreaRect(largest)
                box  = np.int32(cv2.boxPoints(rect)).reshape(4, 2)
                s    = box.sum(axis=1)
                d    = np.diff(box, axis=1)
                TL   = tuple(box[np.argmin(s)])
                BR   = tuple(box[np.argmax(s)])
                TR   = tuple(box[np.argmin(d)])
                BL   = tuple(box[np.argmax(d)])

            x1, y1 = roi_offset
            return {
                'TL': (TL[0] + x1, TL[1] + y1),
                'TR': (TR[0] + x1, TR[1] + y1),
                'BL': (BL[0] + x1, BL[1] + y1),
                'BR': (BR[0] + x1, BR[1] + y1),
            }

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Temporal stability
    # ------------------------------------------------------------------

    def _estimate_velocity(self) -> np.ndarray:
        """
        Estimate the current plate velocity from the recent centre history.

        A short window (last three positions) is used so that the estimate
        responds quickly to changes in the vehicle's approach speed without
        being dominated by a single noisy frame.

        Returns:
            [vx, vy] velocity vector in pixels per frame.
        """
        if len(self.plate_center_history) < 2:
            return np.array([0.0, 0.0])

        recent     = list(self.plate_center_history)[-3:]
        velocities = [np.array(recent[i + 1]) - np.array(recent[i])
                      for i in range(len(recent) - 1)]
        return np.mean(velocities, axis=0)

    def _check_temporal_stability(
        self, corners: Dict[str, Tuple[int, int]]
    ) -> bool:
        """
        Accept or reject a new plate detection based on inter-frame displacement.

        The acceptance threshold is adaptive: it starts at twice the base value
        during a warm-up phase (so the first detections are not rejected), then
        settles to base + velocity-scaled margin once enough history is available.

        Side effects when detection is accepted:
            - ``plate_center_history`` is updated.
            - ``plate_detection_count`` is incremented.
            - ``detection_confidence`` is partially recovered.

        Side effects when detection is rejected:
            - ``detection_confidence`` is decayed by 30 %.

        Args:
            corners: Candidate plate corners for the current frame.

        Returns:
            True if the detection passes the stability check, False otherwise.
        """
        if self.prev_plate_corners is None:
            return True

        xs = [x for x, y in corners.values()]
        ys = [y for x, y in corners.values()]
        current_center = np.array([np.mean(xs), np.mean(ys)])

        prev_xs = [x for x, y in self.prev_plate_corners.values()]
        prev_ys = [y for x, y in self.prev_plate_corners.values()]
        prev_center = np.array([np.mean(prev_xs), np.mean(prev_ys)])

        delta = np.linalg.norm(current_center - prev_center)

        # Use a wider threshold during warm-up to avoid false rejections
        base_threshold = (self.base_max_jump_pixels * 2.0
                          if self.plate_detection_count < self.warmup_frames
                          else self.base_max_jump_pixels)

        # Scale the threshold by current velocity so fast-approaching vehicles
        # are not incorrectly rejected
        expected_movement = np.linalg.norm(self._estimate_velocity())
        threshold = min(base_threshold + expected_movement * 2.0, 200.0)

        if delta > threshold:
            print(f"[AdvancedDetector] Plate jump rejected: "
                  f"{delta:.1f} px > {threshold:.1f} px "
                  f"(estimated velocity = {expected_movement:.1f} px/frame)")
            self.detection_confidence *= 0.7
            return False

        self.detection_confidence = min(1.0, self.detection_confidence * 1.1)
        self.plate_center_history.append(current_center)
        self.plate_detection_count += 1
        return True

    # ------------------------------------------------------------------
    # Plate entry point
    # ------------------------------------------------------------------

    def detect_plate_corners(
        self,
        frame: np.ndarray,
        tail_lights_features: Dict[str, np.ndarray],
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Detect license plate corners using a ROI derived from the tail lights.

        The ROI is positioned below the tail-light bottom points and scaled to
        the inter-light width, so it remains valid across a wide distance range.

        When detection fails or produces an implausible jump, the method falls
        back (in order of preference) to:
            1. A kinematic prediction using the estimated velocity.
            2. The previous frame's corners.
            3. None (if no history is available).

        Args:
            frame:                 BGR input frame.
            tail_lights_features:  Output of ``detect_tail_lights_multifeature``.

        Returns:
            Dict {'TL', 'TR', 'BL', 'BR'} or None.
        """
        height, width = frame.shape[:2]

        bottom_points = tail_lights_features.get('bottom')
        outer_points  = tail_lights_features.get('outer')
        if bottom_points is None or outer_points is None:
            return None

        # Compute the ROI geometry from tail-light geometry
        fari_bottom_y = max(bottom_points[0][1], bottom_points[1][1])
        lights_x_min  = min(outer_points[0][0], outer_points[1][0])
        lights_x_max  = max(outer_points[0][0], outer_points[1][0])
        lights_width  = lights_x_max - lights_x_min

        margin_x = int(lights_width * 0.10)
        x1 = max(0,     int(lights_x_min - margin_x))
        x2 = min(width, int(lights_x_max + margin_x))

        scale = lights_width
        y1 = max(0,      int(fari_bottom_y + 0.15 * scale))
        y2 = min(height, int(fari_bottom_y + 0.45 * scale))

        corners = self._process_plate_in_roi(frame, frame[y1:y2, x1:x2], (x1, y1))

        if corners is None:
            # Detection failed – use kinematic prediction or last known position
            if self.prev_plate_corners is not None and len(self.plate_center_history) >= 2:
                velocity = self._estimate_velocity()
                predicted = {
                    k: (int(px + velocity[0]), int(py + velocity[1]))
                    for k, (px, py) in self.prev_plate_corners.items()
                }
                print("[AdvancedDetector] Plate detection failed – using kinematic prediction.")
                return predicted

            if self.prev_plate_corners is not None:
                print("[AdvancedDetector] Plate detection failed – using previous position.")
                return self.prev_plate_corners

            return None

        if not self._check_temporal_stability(corners):
            if self.prev_plate_corners is not None:
                if self.detection_confidence < self.min_confidence:
                    # Confidence is too low to trust history; accept new detection
                    print(f"[AdvancedDetector] Low confidence ({self.detection_confidence:.2f}) "
                          f"– accepting new plate detection.")
                    self.prev_plate_corners    = corners
                    self.detection_confidence  = 0.5
                    return corners
                return self.prev_plate_corners

        self.prev_plate_corners = corners
        return corners

    def update_plate_bottom_only(
        self,
        frame: np.ndarray,
        tail_lights_features: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Return only the plate's bottom edge [BL, BR] as a (2, 2) float32 array.

        This is a convenience wrapper for the PnP solver, which only needs the
        bottom edge for its plate-bottom correspondence point.

        Args:
            frame:                BGR input frame.
            tail_lights_features: Output of ``detect_tail_lights_multifeature``.

        Returns:
            (2, 2) float32 array [[BL_x, BL_y], [BR_x, BR_y]], or the last
            known plate bottom if detection fails.
        """
        corners = self.detect_plate_corners(frame, tail_lights_features)
        if corners is None:
            return self.prev_plate_bottom

        plate_bottom = np.array(
            [corners['BL'], corners['BR']], dtype=np.float32
        )
        self.prev_plate_bottom = plate_bottom
        return plate_bottom

    # ------------------------------------------------------------------
    # Combined detection
    # ------------------------------------------------------------------

    def detect_all_multifeature(self, frame: np.ndarray) -> Optional[VehicleKeypoints]:
        """
        Run the full detection pipeline and return all keypoints in one call.

        Tail lights are detected first; the plate ROI is derived from their
        geometry. A secondary displacement check (50 px hard limit) discards
        plate detections that jump unreasonably between the detector and the
        tracker even if they pass the adaptive threshold.

        Args:
            frame: BGR input frame.

        Returns:
            VehicleKeypoints instance, or None if tail-light detection fails.
        """
        features_dict, templates_dict = self.detect_tail_lights_with_templates(frame)
        if features_dict is None:
            return None

        plate_corners = self.detect_plate_corners(frame, features_dict)

        plate_bottom = None
        if plate_corners:
            BL = np.array(plate_corners['BL'], dtype=np.float32)
            BR = np.array(plate_corners['BR'], dtype=np.float32)
            plate_bottom = np.array([BL, BR], dtype=np.float32)

            xs = [x for x, y in plate_corners.values()]
            ys = [y for x, y in plate_corners.values()]
            plate_center = (int(np.mean(xs)), int(np.mean(ys)))

            # Secondary check: hard-limit displacement to 50 px
            if len(self.plate_center_history) > 0:
                prev_p = self.plate_center_history[-1]
                if np.linalg.norm(np.array(plate_center) - prev_p) > 50:
                    plate_corners = None   # discard implausible detection
                    confidence    = 0.1
                else:
                    confidence = self.detection_confidence
            else:
                confidence = self.detection_confidence

            self.plate_center_history.append(np.array(plate_center))
        else:
            # Fall back to a rough estimate below the outer-light midpoint
            outer = features_dict['outer']
            plate_center = (
                int((outer[0][0] + outer[1][0]) / 2),
                int((outer[0][1] + outer[1][1]) / 2) + 50,
            )
            confidence = 0.5

        return VehicleKeypoints(
            tail_lights_features=features_dict,
            plate_corners=plate_corners,
            plate_bottom=plate_bottom,
            plate_center=plate_center,
            confidence=confidence,
            templates=templates_dict,
        )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def detect_tail_lights(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return only the outer tail-light pair as a (2, 2) array.

        This wrapper is retained for compatibility with code written against
        the single-feature API. New code should use
        ``detect_tail_lights_multifeature`` directly.

        Args:
            frame: BGR input frame.

        Returns:
            (2, 2) float32 array [[left_x, left_y], [right_x, right_y]], or None.
        """
        features_dict = self.detect_tail_lights_multifeature(frame)
        if features_dict is None:
            return None
        return features_dict.get('outer')