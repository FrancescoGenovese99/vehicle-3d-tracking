"""
redetection.py

Re-detection manager for recovering tail-light tracks after tracker failure.

When the CSRT-based LightTrackers cannot update for several consecutive frames
(as determined by LightTracker.needs_redetection()), the pipeline delegates
to this module to run a fresh detection pass and re-anchor the trackers to the
newly detected light positions.

Kalman-predicted search region
-------------------------------
A 2-state Kalman filter (position + velocity, constant-velocity motion model)
is maintained for the outer light pair. At re-detection time, the filter's
predicted position is used to define a restricted search region: instead of
scanning the entire frame, the detector is run only within a window centred
on the prediction. This dramatically reduces the probability of latching onto
an unrelated bright blob (another vehicle, a streetlight, etc.).

The filter is updated at every frame in which normal tracking succeeds, so
its velocity estimate reflects the true recent motion of the vehicle even
when re-detection is not active.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict

from src.detection.light_detector import LightDetector, LightCandidate
from src.detection.candidate_selector import CandidateSelector


class RedetectionManager:
    """
    Runs a fresh tail-light detection pass when the primary trackers fail.

    The class wraps a LightDetector and a CandidateSelector, and optionally
    augments the detection with a Kalman filter that predicts where the lights
    should be based on their recent motion history.
    """

    def __init__(
        self,
        detector:  LightDetector,
        selector:  CandidateSelector,
        config:    Dict,
    ):
        """
        Args:
            detector: LightDetector instance used for the re-detection pass.
            selector: CandidateSelector that picks the best light pair from
                      the detected candidates.
            config:   Full configuration dict loaded from detection_params.yaml.
                      The relevant sub-key is ``redetection``, which may contain:
                        confidence_threshold    (float) – minimum selector score
                        enable_kalman_prediction (bool) – whether to use Kalman
        """
        self.detector = detector
        self.selector = selector

        redetect_cfg             = config.get('redetection', {})
        self.confidence_threshold = redetect_cfg.get('confidence_threshold', 0.6)
        self.enable_kalman        = redetect_cfg.get('enable_kalman_prediction', True)

        # Kalman filters are initialised lazily on the first update_kalman() call.
        self.kalman_filters: Optional[list] = None
        if self.enable_kalman:
            self._init_kalman_filters()

    # ------------------------------------------------------------------
    # Kalman filter management
    # ------------------------------------------------------------------

    def _init_kalman_filters(self) -> None:
        """
        Initialise two independent Kalman filters — one per light.

        State vector: [x, y, vx, vy]  (position + velocity)
        Measurement:  [x, y]           (position only)

        The constant-velocity transition model (F) assumes the light moves
        by (vx, vy) pixels per frame. Process and measurement noise covariances
        are set to conservative values appropriate for urban vehicle speeds at
        the distances observed in the reference video.
        """
        self.kalman_filters = []

        for _ in range(2):  # one filter per light (left, right)
            kf = cv2.KalmanFilter(4, 2)  # 4 state dims, 2 measurement dims

            # State transition matrix F (constant-velocity model):
            # x(k+1) = x(k) + vx(k),   vx(k+1) = vx(k)  (and similarly for y)
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

            # Measurement matrix H: we observe position only, not velocity.
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ], dtype=np.float32)

            # Process noise Q: small value → trust the motion model more.
            # Increased slightly from the default to allow the filter to adapt
            # when the vehicle changes speed or direction.
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

            # Measurement noise R: reflects typical pixel-level tracker noise.
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

            # Initial posterior error covariance (identity → uncertain start).
            kf.errorCovPost = np.eye(4, dtype=np.float32)

            self.kalman_filters.append(kf)

    # ------------------------------------------------------------------
    # Kalman update / predict
    # ------------------------------------------------------------------

    def update_kalman(
        self,
        centers: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> None:
        """
        Feed the current measured positions into the Kalman filters.

        This should be called once per frame whenever normal tracking succeeds,
        so the velocity estimate stays up to date.

        Args:
            centers: ((left_x, left_y), (right_x, right_y)) — the current
                     outer keypoint positions as tracked by the CSRT layer.
        """
        if not self.enable_kalman or self.kalman_filters is None:
            return

        for i, center in enumerate(centers):
            measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
            self.kalman_filters[i].correct(measurement)

    def predict_position(
        self,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Predict where the lights will be in the next frame.

        Runs the Kalman prediction step (without a measurement update) and
        returns the predicted positions. Used to bias the re-detection search
        region toward the expected light location.

        Returns:
            ((left_x, left_y), (right_x, right_y)) predicted integer positions,
            or None if Kalman prediction is disabled.
        """
        if not self.enable_kalman or self.kalman_filters is None:
            return None

        predicted_centers = []
        for kf in self.kalman_filters:
            pred = kf.predict()
            predicted_centers.append((int(pred[0]), int(pred[1])))

        return tuple(predicted_centers)

    # ------------------------------------------------------------------
    # Re-detection
    # ------------------------------------------------------------------

    def redetect(
        self,
        frame:               np.ndarray,
        last_known_centers:  Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        search_region_scale: float = 2.0,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Attempt to re-detect the tail lights after a tracking failure.

        Pipeline
        --------
        1. Run the Kalman prediction step to obtain a position hint.
        2. Run LightDetector on the full frame to collect all bright-red blobs.
        3. If a reference position is available (Kalman prediction or last known
           centers), filter candidates to those within 150 px of the reference.
        4. Run CandidateSelector to pick the best left–right pair.
        5. Update the Kalman filter with the newly confirmed positions.

        Args:
            frame:               Current BGR frame.
            last_known_centers:  Last successfully tracked positions, used as
                                 a fallback search hint if Kalman is unavailable.
            search_region_scale: Not used in the current implementation (kept
                                 for API compatibility with the ROI variant).

        Returns:
            ((left_x, left_y), (right_x, right_y)) if re-detection succeeds,
            or None if no valid pair is found.
        """
        print("[RedetectionManager] Re-detection triggered.")

        # Step 1: Kalman position prediction
        predicted = None
        if self.enable_kalman:
            predicted = self.predict_position()
            if predicted:
                print(f"  Kalman prediction: {predicted}")

        # Step 2: Full-frame detection
        candidates, mask = self.detector.detect_tail_lights(frame)

        if not candidates:
            print("  No candidates found — re-detection failed.")
            return None

        print(f"  Found {len(candidates)} candidate blobs.")

        # Step 3: Spatial filtering around the reference position
        if last_known_centers or predicted:
            reference = predicted if predicted else last_known_centers
            candidates = self.selector.filter_by_previous_position(
                candidates, reference, max_distance=150
            )
            print(f"  Retained {len(candidates)} candidates within search region.")

        # Step 4: Pair selection
        pair = self.selector.select_tail_light_pair(candidates)

        if pair is None:
            print("  No valid pair found — re-detection failed.")
            return None

        centers = (pair[0].center, pair[1].center)
        print(f"  Re-detection succeeded: {centers}")

        # Step 5: Update Kalman with confirmed positions
        if self.enable_kalman:
            self.update_kalman(centers)

        return centers

    def redetect_with_roi(
        self,
        frame: np.ndarray,
        roi:   Tuple[int, int, int, int],
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Run re-detection restricted to a pre-defined Region of Interest.

        This variant is more efficient than full-frame detection when the
        expected light position is tightly constrained (e.g. immediately after
        a single-frame tracker failure, before the Kalman uncertainty has grown).

        Args:
            frame: Full BGR frame.
            roi:   (x, y, width, height) bounding box of the search region.

        Returns:
            ((left_x, left_y), (right_x, right_y)) in global frame coordinates,
            or None if detection failed within the ROI.
        """
        x, y, w, h = roi
        roi_frame   = frame[y:y + h, x:x + w]

        candidates, _ = self.detector.detect_tail_lights(roi_frame)
        if not candidates:
            return None

        # Translate ROI-local coordinates back to global frame coordinates.
        for candidate in candidates:
            cx, cy           = candidate.center
            candidate.center = (cx + x, cy + y)

        pair = self.selector.select_tail_light_pair(candidates)
        if pair is None:
            return None

        return (pair[0].center, pair[1].center)

    # ------------------------------------------------------------------
    # Search ROI computation
    # ------------------------------------------------------------------

    def compute_search_roi(
        self,
        last_centers: Tuple[Tuple[int, int], Tuple[int, int]],
        frame_shape:  Tuple[int, int],
        scale:        float = 2.0,
    ) -> Tuple[int, int, int, int]:
        """
        Compute a search ROI centred on the last known light positions.

        The ROI dimensions are proportional to the current apparent inter-light
        distance, scaled by ``scale``. The resulting box is clamped to the
        frame boundaries.

        Args:
            last_centers: Last accepted ((lx, ly), (rx, ry)) positions.
            frame_shape:  (height, width) of the video frame.
            scale:        Multiplier applied to the inter-light distance to
                          determine the ROI half-size.

        Returns:
            (x, y, width, height) of the search ROI.
        """
        left, right = last_centers

        center_x = (left[0]  + right[0]) // 2
        center_y = (left[1]  + right[1]) // 2
        distance = abs(right[0] - left[0])

        roi_w = int(distance * scale * 1.5)
        roi_h = int(distance * scale)

        x = max(0, center_x - roi_w // 2)
        y = max(0, center_y - roi_h // 2)
        w = min(roi_w, frame_shape[1] - x)
        h = min(roi_h, frame_shape[0] - y)

        return (x, y, w, h)