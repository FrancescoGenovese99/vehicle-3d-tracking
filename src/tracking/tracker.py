"""
tracker.py

Frame-by-frame tracking of tail-light keypoints using OpenCV correlation trackers.

Each LightTracker instance manages a pair of OpenCV trackers — one per keypoint
(e.g. the two outer edges, or the two top corners) — for a single feature group.
Three separate LightTracker objects are therefore used by the pipeline: one for
the 'outer', one for the 'top', and one for the 'bottom' group.

Tracker type is configurable in detection_params.yaml:
    CSRT  – Discriminative Correlation Filter with Channel and Spatial Reliability.
            Recommended: most accurate in the presence of partial occlusion and
            illumination changes, at the cost of higher per-frame CPU time.
    KCF   – Kernelised Correlation Filter. Faster than CSRT, less robust.
    MOSSE – Minimum Output Sum of Squared Error. Fastest; suitable only when the
            scene has high contrast and motion is small.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum


class TrackerType(Enum):
    """Enumeration of supported OpenCV tracker backends."""
    CSRT  = "CSRT"
    KCF   = "KCF"
    MOSSE = "MOSSE"


class LightTracker:
    """
    Tracks a pair of keypoints across consecutive video frames.

    The tracker wraps two OpenCV single-object trackers, one per keypoint.
    Internally it converts each point to a small square bounding box (whose
    half-side is ``bbox_padding`` pixels) to feed into the OpenCV API, and
    converts the output bounding box back to a centre point.

    Failure handling
    ----------------
    If one of the two internal trackers fails on a given frame, the last known
    centre for that point is reused and the frame is still considered a success,
    provided the other tracker succeeded. Only when both trackers fail is the
    frame declared a failure and ``frames_since_detection`` incremented.
    """

    def __init__(self, config: Dict):
        """
        Initialise the tracker from the pipeline configuration.

        Args:
            config: Full configuration dict loaded from detection_params.yaml.
                    The relevant sub-key is ``tracking``, which must contain:
                      tracker_type  (str)  – 'CSRT', 'KCF', or 'MOSSE'
                      bbox_padding  (int)  – half-side of the tracking window (px)
                      max_frames_lost (int)– frames before re-detection is flagged
        """
        tracking_cfg = config.get('tracking', {})

        tracker_name     = tracking_cfg.get('tracker_type', 'CSRT')
        self.tracker_type = TrackerType[tracker_name]

        # Half-side of the square bounding box built around each keypoint.
        # A value of 20 px gives a 40×40 px tracking window, which is sufficient
        # for tail-light blobs at the distances of interest (5–30 m).
        self.bbox_padding     = tracking_cfg.get('bbox_padding', 20)

        # Number of consecutive frames without a successful update before
        # needs_redetection() returns True.
        self.max_frames_lost  = tracking_cfg.get('max_frames_lost', 10)

        # --- Internal state ---
        self.trackers              = []    # list of active OpenCV Tracker objects
        self.current_centers       = None  # last accepted centre positions
        self.frames_since_detection = 0
        self.is_initialized        = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_tracker(self) -> cv2.Tracker:
        """
        Instantiate a fresh OpenCV tracker of the configured type.

        Returns:
            A new, uninitialised OpenCV Tracker object.

        Raises:
            ValueError: If the tracker type is not one of the supported values.
        """
        if self.tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == TrackerType.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")

    def _point_to_bbox(self, point: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Build a square bounding box centred on a keypoint.

        The box is clamped to the frame origin to avoid negative coordinates
        (the upper-right clamp is handled by OpenCV itself).

        Args:
            point: (x, y) pixel coordinate of the keypoint centre.

        Returns:
            (x, y, w, h) OpenCV bounding box; w == h == 2 * bbox_padding.
        """
        x, y      = point
        half_size = self.bbox_padding
        return (
            max(0, x - half_size),
            max(0, y - half_size),
            2 * half_size,
            2 * half_size,
        )

    def _bbox_to_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Extract the centre pixel from an OpenCV bounding box.

        Args:
            bbox: (x, y, w, h) as returned by tracker.update().

        Returns:
            (cx, cy) integer centre coordinates.
        """
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        frame: np.ndarray,
        initial_points: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> None:
        """
        Start tracking from a pair of detected keypoint positions.

        Creates one OpenCV tracker per point, initialises each on the
        corresponding bounding box patch within ``frame``, and marks the
        tracker as ready for subsequent update() calls.

        Args:
            frame:          First BGR frame in the tracking sequence.
            initial_points: ((left_x, left_y), (right_x, right_y)) — the
                            pixel coordinates of the two keypoints to track.
        """
        self.trackers = []

        for point in initial_points:
            tracker = self._create_tracker()
            bbox    = self._point_to_bbox(point)
            tracker.init(frame, bbox)
            self.trackers.append(tracker)

        self.current_centers        = initial_points
        self.frames_since_detection = 0
        self.is_initialized         = True

        print(f"[LightTracker] Initialised with {len(self.trackers)} keypoints "
              f"({self.tracker_type.value})")

    def update(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Advance the trackers by one frame.

        Each internal OpenCV tracker is updated independently. If at least one
        tracker succeeds, the method returns True and the updated centre
        positions. A failed tracker reuses its last known position rather than
        returning None, so the caller always receives a complete point pair when
        the overall result is True.

        If all trackers fail, the method returns False and the last accepted
        positions (so the caller can decide how to handle the failure).

        Args:
            frame: Current BGR frame.

        Returns:
            (success, centers) where ``success`` is True if at least one
            internal tracker succeeded, and ``centers`` is a tuple of two
            (x, y) integer coordinates (or None if the tracker has never
            been initialised).
        """
        if not self.is_initialized:
            return False, None

        updated_centers = []
        success_count   = 0

        for i, tracker in enumerate(self.trackers):
            ok, bbox = tracker.update(frame)

            if ok:
                center = self._bbox_to_point(bbox)
                updated_centers.append(center)
                success_count += 1
            else:
                # Tracker failed for this point: reuse the last known position
                # so the output tuple always has exactly two entries.
                if self.current_centers and i < len(self.current_centers):
                    updated_centers.append(self.current_centers[i])

        if success_count > 0:
            # At least one tracker is still alive: accept the update.
            self.current_centers        = tuple(updated_centers)
            self.frames_since_detection = 0
            return True, self.current_centers
        else:
            # Both trackers failed.
            self.frames_since_detection += 1
            return False, self.current_centers

    def reinitialize(
        self,
        frame: np.ndarray,
        new_points: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> None:
        """
        Reset and restart tracking at new positions (called after re-detection).

        Equivalent to a fresh initialize() call; the previous tracker state is
        discarded entirely.

        Args:
            frame:      Current BGR frame.
            new_points: New keypoint positions returned by the re-detector.
        """
        print("[LightTracker] Re-initialising after re-detection.")
        self.initialize(frame, new_points)

    def needs_redetection(self) -> bool:
        """
        Check whether the tracker has been lost long enough to require a full
        re-detection pass.

        Returns:
            True if ``frames_since_detection`` exceeds ``max_frames_lost``.
        """
        return self.frames_since_detection > self.max_frames_lost

    def get_current_centers(
        self,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Return the most recently accepted keypoint positions.

        Returns:
            Tuple of two (x, y) coordinates, or None if never initialised.
        """
        return self.current_centers

    def reset(self) -> None:
        """
        Fully reset the tracker, discarding all internal state.

        Called when the vehicle is lost beyond recovery and the pipeline is
        preparing for a fresh detection attempt.
        """
        self.trackers               = []
        self.current_centers        = None
        self.frames_since_detection = 0
        self.is_initialized         = False