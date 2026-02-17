"""
camera_calibration.py - Camera calibration from chessboard images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import glob


# Pattern sizes tried by auto-detection, roughly from most to least specific.
# (7, 3) works well when the board is held at a steep angle or only partially
# visible, since only a horizontal strip of inner corners needs to be in frame.
_CANDIDATE_PATTERNS = [
    (9, 6), (7, 7), (7, 5), (7, 4), (7, 3), (6, 5), (6, 4), (5, 4), (5, 3),
]


def _preprocess_for_chessboard(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, equalise the histogram and apply a Gaussian blur.
    This significantly improves corner detection on low-contrast wooden boards.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray


def detect_pattern_size(image_paths: List[str], min_success: int = 3) -> Optional[Tuple[int, int]]:
    """
    Try a list of candidate pattern sizes and return the first one that
    successfully detects corners in at least `min_success` images.

    Useful when you are unsure of the exact inner-corner count of the board.

    Args:
        image_paths: Paths to calibration images.
        min_success: Minimum number of images that must succeed.

    Returns:
        Best (cols, rows) pattern size, or None if nothing worked.
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )

    for pat in _CANDIDATE_PATTERNS:
        n_ok = 0
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = _preprocess_for_chessboard(img)
            ret, _ = cv2.findChessboardCorners(gray, pat, flags)
            if ret:
                n_ok += 1
        if n_ok >= min_success:
            print(f"  Auto-detected pattern size: {pat}  ({n_ok}/{len(image_paths)} images)")
            return pat

    return None


class CameraCalibrator:
    """
    Camera calibration from a set of chessboard images.

    Usage::

        cal = CameraCalibrator(pattern_size=(7, 3), square_size=0.025)
        for path in image_paths:
            cal.add_image(path)
        K, dist, err = cal.calibrate()
        cal.save_calibration("data/calibration/camera1.npz", K, dist)

    Pattern size note
    -----------------
    ``pattern_size`` is the number of *inner* corners (cols, rows), not the
    number of squares.  A standard 8×8 chess board has 7×7 inner corners, but
    if the board is always held at an angle or only partially visible, using a
    smaller strip like (7, 3) is more reliable because OpenCV only needs to
    locate that many corners in the image.
    """

    # Detection flags tuned for low-contrast / wooden chessboards.
    # FILTER_QUADS helps reject false positives from wood grain.
    _DETECTION_FLAGS = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )

    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        Args:
            pattern_size: Number of inner corners (cols, rows).
                          For a standard 8×8 board held at an angle, (7, 3)
                          is a robust choice (verified on the project images).
            square_size:  Physical side length of one square in metres.
        """
        self.pattern_size = pattern_size
        self.square_size  = square_size

        # 3-D object points for one board view (Z = 0, flat board)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = (
            np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        )
        self.objp *= square_size

        self.objpoints: List[np.ndarray] = []   # 3-D world points per image
        self.imgpoints: List[np.ndarray] = []   # 2-D image points per image
        self.image_size: Optional[Tuple[int, int]] = None

    def find_corners(
        self, image: np.ndarray, visualize: bool = False
    ) -> Optional[np.ndarray]:
        """
        Detect and sub-pixel refine chessboard corners in a BGR image.

        The image is pre-processed (histogram equalisation + blur) before
        detection — this is essential for low-contrast wooden boards.

        Args:
            image:     BGR input image.
            visualize: Show detected corners for 500 ms (requires a display).

        Returns:
            (N, 1, 2) float32 corner array, or None if not found.
        """
        gray = _preprocess_for_chessboard(image)

        ret, corners = cv2.findChessboardCorners(
            gray, self.pattern_size, self._DETECTION_FLAGS
        )

        if not ret:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if visualize:
            vis = image.copy()
            cv2.drawChessboardCorners(vis, self.pattern_size, corners, ret)
            cv2.imshow("Chessboard corners", vis)
            cv2.waitKey(500)

        return corners

    def add_image(self, image_path: str, visualize: bool = False) -> bool:
        """
        Add one calibration image to the dataset.

        Args:
            image_path: Path to a JPEG or PNG image.
            visualize:  Show detected corners while processing.

        Returns:
            True if corners were found, False otherwise.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Cannot load: {image_path}")
            return False

        if self.image_size is None:
            self.image_size = (img.shape[1], img.shape[0])

        corners = self.find_corners(img, visualize)
        name = Path(image_path).name

        if corners is not None:
            self.objpoints.append(self.objp.copy())
            self.imgpoints.append(corners)
            print(f"  ✓ {name}")
            return True

        print(f"  ✗ {name} — pattern not found")
        return False

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run calibration on all images added so far.

        Returns:
            (camera_matrix, dist_coeffs, mean_reprojection_error_px)

        Raises:
            ValueError: Fewer than 3 usable images available.
        """
        if len(self.objpoints) < 3:
            raise ValueError(
                f"Need at least 3 usable images, got {len(self.objpoints)}."
            )

        print(f"\nCalibrating with {len(self.objpoints)} images...")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )

        total_error = 0.0
        for i in range(len(self.objpoints)):
            proj, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], K, dist
            )
            total_error += cv2.norm(self.imgpoints[i], proj, cv2.NORM_L2) / len(proj)

        mean_error = total_error / len(self.objpoints)

        print(f"  RMS error              : {ret:.4f}")
        print(f"  Mean reprojection error: {mean_error:.4f} px")

        return K, dist, mean_error

    def save_calibration(
        self,
        output_path: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ):
        """
        Save calibration data to a .npz file with keys:
        'camera_matrix', 'dist_coefficients', 'image_size'.

        Args:
            output_path:   Destination .npz path (parent dirs created automatically).
            camera_matrix: 3×3 intrinsic matrix.
            dist_coeffs:   Distortion coefficient vector.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            out,
            camera_matrix=camera_matrix,
            dist_coefficients=dist_coeffs,
            image_size=np.array(self.image_size),
        )
        print(f"  Calibration saved to: {out}")

    @staticmethod
    def calibrate_from_images(
        image_pattern: str,
        pattern_size: Tuple[int, int],
        square_size: float,
        output_path: str,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convenience wrapper: calibrate directly from a glob pattern and save.

        Args:
            image_pattern: Glob string, e.g. ``"data/calibration/images/*.jpg"``.
            pattern_size:  Inner corner count (cols, rows).
            square_size:   Square side length in metres.
            output_path:   Destination .npz file.
            visualize:     Show detected corners while processing.

        Returns:
            (camera_matrix, dist_coeffs, mean_error)
        """
        image_files = glob.glob(image_pattern)
        if not image_files:
            raise FileNotFoundError(f"No images found matching: {image_pattern}")

        print(f"Found {len(image_files)} calibration images.")

        calibrator = CameraCalibrator(pattern_size, square_size)
        for path in sorted(image_files):
            calibrator.add_image(path, visualize)

        K, dist, mean_error = calibrator.calibrate()
        calibrator.save_calibration(output_path, K, dist)

        if visualize:
            cv2.destroyAllWindows()

        return K, dist, mean_error


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Remove lens distortion from an image and crop to the valid pixel region.

    Args:
        image:         Distorted BGR image.
        camera_matrix: 3×3 intrinsic matrix.
        dist_coeffs:   Distortion coefficient vector.

    Returns:
        Undistorted and cropped BGR image.
    """
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_K)

    x, y, rw, rh = roi
    return undistorted[y:y + rh, x:x + rw]