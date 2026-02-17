"""
candidate_selector.py

Selects the most likely pair of tail lights from a list of detected light
candidates. Designed for rear-vehicle detection on a Toyota Aygo X, but the
scoring logic is general enough to be reused for other vehicles.

Pipeline overview
-----------------
1. Soft filtering  – removes only obvious outliers (border blobs, tiny noise).
2. Pair scoring    – evaluates every combination on four weighted criteria.
3. Best-pair pick  – returns the pair with the highest composite score.
4. Tracking aids   – helpers for position-based filtering and missing-light
                     estimation across consecutive frames.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .light_detector import LightCandidate


class CandidateSelector:
    """
    Selects the most probable tail-light pair from a set of detected candidates.

    Scoring weights (see ``compute_pair_score`` for details):
        - Size score       35 %  – larger blobs are closer / more relevant
        - Vertical score   35 %  – real tail lights share nearly the same Y
        - Horizontal score 20 %  – pair must be plausibly separated
        - Area similarity  10 %  – both lights should be roughly the same size
    """

    def __init__(self, config: Dict, frame_width: int, frame_height: int = 1080):
        """
        Args:
            config:       Full configuration dictionary (from detection_params.yaml).
            frame_width:  Frame width in pixels.
            frame_height: Frame height in pixels (default 1080).
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        sel_cfg = config.get('tail_lights_selection', {})
        self.min_horizontal_distance = sel_cfg.get('min_horizontal_distance', 50)
        self.max_horizontal_distance_ratio = sel_cfg.get('max_horizontal_distance_ratio', 0.8)
        self.max_vertical_offset = sel_cfg.get('max_vertical_offset', 60)
        self.min_area_similarity = sel_cfg.get('min_area_similarity', 0.4)
        self.min_pair_score = sel_cfg.get('min_pair_score', 0.25)

        print(f"[CandidateSelector] Initialized: {frame_width}x{frame_height}")
        print(f"  Min horizontal distance : {self.min_horizontal_distance} px")
        print(f"  Max vertical offset     : {self.max_vertical_offset} px")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def soft_filter_candidates(self, candidates: List[LightCandidate]) -> List[LightCandidate]:
        """
        Remove obvious outliers while keeping intentionally permissive thresholds
        so that faint or distant lights are not discarded prematurely.

        Rejection criteria:
            - Center within 30 px of the left or right frame edge.
            - Center above 10 % or below 95 % of frame height.
            - Blob area < 50 px² (sensor noise / compression artifacts).
            - Circularity < 0.05 (degenerate / broken contour).
        """
        filtered = []

        for c in candidates:
            cx, cy = c.center

            # Discard blobs touching the lateral frame borders
            if cx < 30 or cx > self.frame_width - 30:
                continue

            # Keep only blobs in a reasonable vertical band of the frame
            y_ratio = cy / self.frame_height
            if y_ratio < 0.10 or y_ratio > 0.95:
                continue

            # Discard microscopic blobs (noise)
            if c.area < 50:
                continue

            # Discard nearly-degenerate contours
            if c.circularity < 0.05:
                continue

            filtered.append(c)

        return filtered

    # ------------------------------------------------------------------
    # Pair scoring
    # ------------------------------------------------------------------

    def compute_pair_score(
        self, c1: LightCandidate, c2: LightCandidate
    ) -> Tuple[float, Dict]:
        """
        Compute a composite score in [0, 1] for a candidate pair.

        The score is adaptive: it works both when the vehicle is close
        (large blobs, wide horizontal spread) and when it is far away
        (small blobs, narrow spread), without requiring manual threshold
        tuning for each distance regime.

        Args:
            c1: Left / first candidate.
            c2: Right / second candidate.

        Returns:
            total_score: Weighted composite score.
            metrics:     Dictionary with all intermediate values (useful for
                         debugging and logging).
        """
        dx = abs(c1.center[0] - c2.center[0])
        dy = abs(c1.center[1] - c2.center[1])
        area_ratio = min(c1.area, c2.area) / max(c1.area, c2.area)
        avg_area = (c1.area + c2.area) / 2

        # --- Score 1: Size (distance proxy) ---
        # Blob area correlates with vehicle distance:
        #   >= 400 px²  → very close    (score 1.0)
        #   200–400 px² → close-medium  (score 0.7–1.0)
        #   100–200 px² → medium-far    (score 0.4–0.7)
        #   <  100 px²  → far / noise   (score 0.1–0.4)
        if avg_area >= 400:
            size_score = 1.0
        elif avg_area >= 200:
            size_score = 0.7 + (avg_area - 200) / 200 * 0.3
        elif avg_area >= 100:
            size_score = 0.4 + (avg_area - 100) / 100 * 0.3
        else:
            size_score = max(0.1, avg_area / 100 * 0.3)

        # --- Score 2: Vertical alignment ---
        # Genuine tail lights lie on the same horizontal line.
        # Score decays linearly to 0 as dy approaches max_vertical_offset.
        vertical_score = max(0.0, 1.0 - (dy / self.max_vertical_offset))

        # --- Score 3: Horizontal separation ---
        # Rewards plausible inter-light distances:
        #   >= 200 px → wide spread (vehicle is close)
        #   100–200 px → medium spread
        #   min–100 px → narrow spread (vehicle is far)
        horizontal_score = 0.0
        max_dx = self.frame_width * self.max_horizontal_distance_ratio
        if self.min_horizontal_distance <= dx <= max_dx:
            if dx >= 200:
                horizontal_score = min(1.0, 0.8 + (dx - 200) / 300 * 0.2)
            elif dx >= 100:
                horizontal_score = 0.6 + (dx - 100) / 100 * 0.2
            else:
                horizontal_score = 0.4 + (dx - self.min_horizontal_distance) / 50 * 0.2

        # --- Score 4: Area similarity ---
        # A symmetric pair should have similar blob sizes.
        area_score = area_ratio  # already in [0, 1]

        # --- Weighted sum ---
        weights = {
            'size':       0.35,
            'vertical':   0.35,
            'horizontal': 0.20,
            'area':       0.10,
        }

        total_score = (
            weights['size']       * size_score
            + weights['vertical']   * vertical_score
            + weights['horizontal'] * horizontal_score
            + weights['area']       * area_score
        )

        metrics = {
            'dx':               dx,
            'dy':               dy,
            'area_ratio':       area_ratio,
            'avg_area':         avg_area,
            'size_score':       size_score,
            'vertical_score':   vertical_score,
            'horizontal_score': horizontal_score,
            'area_score':       area_score,
            'total_score':      total_score,
        }

        return total_score, metrics

    # ------------------------------------------------------------------
    # Main selection
    # ------------------------------------------------------------------

    def select_tail_light_pair(
        self,
        candidates: List[LightCandidate],
        prefer_center: bool = False,
    ) -> Optional[Tuple[LightCandidate, LightCandidate]]:
        """
        Select the best tail-light pair from a list of candidates.

        Args:
            candidates:    Detected light candidates for the current frame.
            prefer_center: When True, add a bonus (up to +15 %) to pairs whose
                           midpoint is close to the horizontal frame center.
                           Useful during initial detection when the vehicle
                           enters the frame from the front.

        Returns:
            A (left, right) tuple ordered by x-coordinate, or None if no pair
            passes the minimum score threshold.
        """
        # Step 1 – remove obvious outliers
        candidates = self.soft_filter_candidates(candidates)

        if len(candidates) < 2:
            return None

        best_pair = None
        best_score = self.min_pair_score  # pairs below this threshold are ignored
        best_metrics = None
        center_x = self.frame_width / 2

        # Step 2 – evaluate every possible pair
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = candidates[i], candidates[j]
                score, metrics = self.compute_pair_score(c1, c2)

                # Step 3 – optional bonus for center-aligned pairs
                if prefer_center:
                    pair_center_x = (c1.center[0] + c2.center[0]) / 2
                    distance_from_center = abs(pair_center_x - center_x) / self.frame_width
                    score += (1.0 - distance_from_center) * 0.15

                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    # Ensure left-to-right ordering
                    best_pair = (c1, c2) if c1.center[0] < c2.center[0] else (c2, c1)

        # Step 4 – log selected pair
        if best_pair and best_metrics:
            left, right = best_pair
            y_avg = (left.center[1] + right.center[1]) / 2
            y_percent = (y_avg / self.frame_height) * 100

            # Human-readable distance classification based on blob area
            avg_area = best_metrics['avg_area']
            if avg_area >= 400:
                distance_class = "VERY CLOSE"
            elif avg_area >= 200:
                distance_class = "CLOSE-MEDIUM"
            elif avg_area >= 100:
                distance_class = "MEDIUM-FAR"
            else:
                distance_class = "FAR"

            print(f"   ✓ Pair selected — Score: {best_score:.3f} ({distance_class})")
            print(f"     dx={best_metrics['dx']:.0f} px, dy={best_metrics['dy']:.0f} px")
            print(f"     Area: {avg_area:.0f} px²,  Y: {y_percent:.1f} % of frame")
            print(
                f"     Scores: SIZE={best_metrics['size_score']:.2f} (35 %), "
                f"VERT={best_metrics['vertical_score']:.2f} (35 %), "
                f"HORIZ={best_metrics['horizontal_score']:.2f} (20 %)"
            )

        return best_pair

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def get_tail_light_centers(
        self,
        candidates: List[LightCandidate],
        prefer_center: bool = False,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Return the pixel centers of the selected tail-light pair.

        Args:
            candidates:    Detected light candidates for the current frame.
            prefer_center: Passed through to ``select_tail_light_pair``.

        Returns:
            ((left_x, left_y), (right_x, right_y)), or None if no pair found.
        """
        pair = self.select_tail_light_pair(candidates, prefer_center=prefer_center)
        if pair is None:
            return None

        left, right = pair
        return left.center, right.center

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------

    def filter_by_previous_position(
        self,
        candidates: List[LightCandidate],
        previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
        max_distance: int = 150,
    ) -> List[LightCandidate]:
        """
        Retain only the candidates that are close to the previously detected
        tail-light positions. Used during temporal tracking to discard
        unrelated light sources that appear between frames.

        Args:
            candidates:        Current-frame candidates.
            previous_centers:  ((left_x, left_y), (right_x, right_y)) from
                               the last successful detection.
            max_distance:      Maximum pixel distance from either previous
                               center for a candidate to be retained.

        Returns:
            Filtered candidate list.
        """
        if not candidates or not previous_centers:
            return candidates

        prev_left, prev_right = previous_centers
        filtered = []

        for candidate in candidates:
            cx, cy = candidate.center
            dist_left  = np.hypot(cx - prev_left[0],  cy - prev_left[1])
            dist_right = np.hypot(cx - prev_right[0], cy - prev_right[1])

            if min(dist_left, dist_right) < max_distance:
                filtered.append(candidate)

        return filtered

    def estimate_missing_light(
        self,
        single_light: LightCandidate,
        previous_centers: Tuple[Tuple[int, int], Tuple[int, int]],
        is_left: bool,
    ) -> Tuple[int, int]:
        """
        Estimate the position of a temporarily occluded tail light using the
        inter-light distance measured in the previous frame.

        Args:
            single_light:      The one light that is still visible.
            previous_centers:  ((left_x, left_y), (right_x, right_y)) from
                               the last successful detection.
            is_left:           True if ``single_light`` is the LEFT light,
                               so the RIGHT one must be estimated; False for
                               the opposite case.

        Returns:
            (x, y) estimated position of the missing light.
        """
        prev_left, prev_right = previous_centers
        prev_distance = prev_right[0] - prev_left[0]  # horizontal span

        if is_left:
            # Single light is on the left → estimate right light position
            estimated_x = single_light.center[0] + prev_distance
        else:
            # Single light is on the right → estimate left light position
            estimated_x = single_light.center[0] - prev_distance

        estimated_y = single_light.center[1]

        return int(estimated_x), int(estimated_y)