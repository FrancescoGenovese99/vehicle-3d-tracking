"""
draw_utils.py - Visualization helpers for detections, tracking, and debug views.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


def draw_tracking_info(frame: np.ndarray,
                       frame_idx: int,
                       method: str,
                       distance: float,
                       num_points: int) -> None:
    """Draws tracking info (frame index, method, distance, point count) in the top-left corner."""
    y_offset = 30
    for text in [f"Frame: {frame_idx}", f"Method: {method}",
                 f"Distance: {distance:.2f}m", f"Points: {num_points}"]:
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25


def draw_motion_type_overlay(frame: np.ndarray,
                              motion_type: str,
                              bg_alpha: float = 0.6) -> None:
    """
    Draws a semi-transparent motion type banner at the bottom of the frame.
    Color coding: TRANSLATION = green, STEERING = red, other = orange.
    """
    h, w = frame.shape[:2]

    bg_colors = {
        "TRANSLATION": (0, 128, 0),
        "STEERING":    (0, 0, 128),
    }
    bg_color = bg_colors.get(motion_type, (0, 82, 128))

    text = f"MOTION: {motion_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 1.2, 3

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = h - 20

    overlay = frame.copy()
    pad = 15
    cv2.rectangle(overlay,
                  (x - pad, y - text_h - pad),
                  (x + text_w + pad, y + baseline + pad),
                  bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def draw_3d_axes(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float = 1.5,
    thickness: int = 4
) -> None:
    """
    Projects and draws the vehicle reference frame axes onto the image.
    Colors: X (forward) = red, Y (left) = green, Z (up) = blue.
    """
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)

    projected, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)

    origin, x_end, y_end, z_end = [tuple(p) for p in projected]

    cv2.line(frame, origin, x_end, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.line(frame, origin, y_end, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(frame, origin, z_end, (255, 0, 0), thickness, cv2.LINE_AA)

    cv2.circle(frame, origin, 8, (255, 255, 255), -1)
    cv2.circle(frame, origin, 10, (0, 0, 0), 2)

    def draw_label_with_bg(img, text, pos, color):
        font, font_scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, text_h), _ = cv2.getTextSize(text, font, font_scale, th)
        cv2.rectangle(img, (pos[0] - 2, pos[1] - text_h - 2),
                      (pos[0] + tw + 2, pos[1] + 2), (0, 0, 0), -1)
        cv2.putText(img, text, pos, font, font_scale, color, th, cv2.LINE_AA)

    draw_label_with_bg(frame, 'X (forward)', x_end, (0, 0, 255))
    draw_label_with_bg(frame, 'Y (left)',    y_end, (0, 255, 0))
    draw_label_with_bg(frame, 'Z (up)',      z_end, (255, 0, 0))
    draw_label_with_bg(frame, 'Origin', (origin[0] + 15, origin[1] - 10), (255, 255, 255))


def draw_bbox_3d(frame: np.ndarray,
                 projected_points: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2) -> None:
    """
    Draws a 3D bounding box from 8 projected vertices (bottom face first, then top).
    The rear face is highlighted in red for orientation clarity.
    """
    if projected_points is None or len(projected_points) != 8:
        return

    pts = projected_points.astype(int)

    for i, j in [(0,1),(1,2),(2,3),(3,0)]:   # bottom face
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    for i, j in [(4,5),(5,6),(6,7),(7,4)]:   # top face
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    for i, j in [(0,4),(1,5),(2,6),(3,7)]:   # vertical edges
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)

    # Highlight rear face in red
    rear = (0, 0, 255)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), rear, thickness + 2)
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), rear, thickness + 2)


def draw_dashed_line(frame, pt1, pt2, color, thickness=1, gap=10):
    """Draws a dashed line between two points by sampling at regular intervals."""
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    if dist < 1:
        return

    pts = [(int((1 - r/dist) * pt1[0] + (r/dist) * pt2[0]),
            int((1 - r/dist) * pt1[1] + (r/dist) * pt2[1]))
           for r in np.arange(0, dist, gap)]

    for i in range(0, len(pts) - 1, 2):
        cv2.line(frame, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


class DrawUtils:
    """Collection of static visualization methods for vanishing points and debug views."""

    @staticmethod
    def draw_vp_convergence(
        frame: np.ndarray,
        features_curr: dict,
        features_prev: Optional[dict],
        vx_motion: Optional[np.ndarray],
        vy_lateral: Optional[np.ndarray],
        show_labels: bool = True
    ):
        """
        Draws vanishing point convergence lines on the frame.

        Vy (lateral VP, cyan)  — solid lines between L/R feature pairs,
                                  dashed lines extending toward Vy.
        Vx (motion VP, orange) — arrows showing point displacement t→t+1,
                                  dashed lines extending toward Vx.

        When a VP is off-screen, an arrow is drawn toward it from the frame center.

        Args:
            features_curr: Detected features for the current frame.
                           Keys: 'outer', 'top', 'plate_bottom' → list of 2D points.
            features_prev: Detected features for the previous frame (needed for Vx).
            vx_motion:     Motion vanishing point [x, y], or None.
            vy_lateral:    Lateral vanishing point [x, y], or None.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        def clamp_vp(vp):
            if vp is None:
                return None
            return (int(np.clip(vp[0], -w*2, w*3)),
                    int(np.clip(vp[1], -h*2, h*3)))

        def is_on_screen(pt, margin=100):
            return -margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin

        def draw_offscreen_arrow(vp_world, color, label_y):
            """Draws an arrow from the frame center pointing toward an off-screen VP."""
            center = np.array([w / 2, h / 2])
            direction = np.array(vp_world, dtype=float) - center
            norm = np.linalg.norm(direction)
            if norm > 1:
                direction /= norm
                arrow_end = (center + direction * 80).astype(int)
                cv2.arrowedLine(overlay, tuple(center.astype(int)),
                                tuple(arrow_end), color, 3, tipLength=0.3)
            cv2.putText(overlay,
                        f"VP→({int(vp_world[0])},{int(vp_world[1])})",
                        (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Lateral VP (Vy) ---
        if vy_lateral is not None:
            vy_pt = clamp_vp(vy_lateral)
            vy_visible = is_on_screen(vy_pt, margin=200)
            color_vy = (255, 220, 0)

            pairs = []
            for key, label in [('outer', 'outer'), ('top', 'top'), ('plate_bottom', 'plate')]:
                pts = features_curr.get(key, [])
                if len(pts) == 2:
                    pairs.append((pts[0], pts[1], label))

            for pL, pR, _ in pairs:
                pL_i = (int(pL[0]), int(pL[1]))
                pR_i = (int(pR[0]), int(pR[1]))
                cv2.line(overlay, pL_i, pR_i, color_vy, 2, cv2.LINE_AA)
                if vy_visible:
                    draw_dashed_line(overlay, pL_i, vy_pt, color_vy, 1, gap=8)
                    draw_dashed_line(overlay, pR_i, vy_pt, color_vy, 1, gap=8)

            if vy_visible:
                cv2.circle(overlay, vy_pt, 12, color_vy, -1)
                cv2.circle(overlay, vy_pt, 15, (255, 255, 255), 2)
                if show_labels:
                    cv2.putText(overlay, "Vy (lateral)",
                                (vy_pt[0] + 20, vy_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vy, 2, cv2.LINE_AA)
            else:
                draw_offscreen_arrow(vy_lateral, color_vy, label_y=h - 90)

        # --- Motion VP (Vx) ---
        if vx_motion is not None and features_prev is not None:
            vx_pt = clamp_vp(vx_motion)
            vx_visible = is_on_screen(vx_pt, margin=200)
            color_vx = (0, 140, 255)

            for key in ['outer', 'top']:
                curr = features_curr.get(key, [])
                prev = features_prev.get(key, [])
                for i in range(min(len(curr), len(prev))):
                    p_prev = (int(prev[i][0]), int(prev[i][1]))
                    p_curr = (int(curr[i][0]), int(curr[i][1]))
                    if np.linalg.norm(np.array(p_curr) - np.array(p_prev)) < 1.0:
                        continue
                    cv2.arrowedLine(overlay, p_prev, p_curr, color_vx, 2,
                                    tipLength=0.2, line_type=cv2.LINE_AA)
                    if vx_visible:
                        draw_dashed_line(overlay, p_curr, vx_pt, color_vx, 1, gap=8)

            if vx_visible:
                cv2.circle(overlay, vx_pt, 12, color_vx, -1)
                cv2.circle(overlay, vx_pt, 15, (255, 255, 255), 2)
                if show_labels:
                    cv2.putText(overlay, "Vx (motion)",
                                (vx_pt[0] + 20, vx_pt[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vx, 2, cv2.LINE_AA)
            else:
                draw_offscreen_arrow(vx_motion, color_vx, label_y=h - 65)

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    @staticmethod
    def create_debug_mask_frame(
        frame: np.ndarray,
        detector,
        features_dict: Optional[dict],
        plate_bottom: Optional[np.ndarray],
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        frame_idx: int = 0
    ) -> np.ndarray:
        """
        Builds a black debug frame showing the intermediate processing steps:
          - Red   : HSV light mask
          - Green : Final plate contour (after gradient + morphology pipeline)
          - Cyan  : Top light anchor points
          - Green : Outer light anchor points (PnP reference)
          - Yellow: Bottom light anchor points
          - Magenta: License plate bottom edge (BL, BR)
          - Orange: Projected 3D origin (rear axle center at ground level)

        The plate contour pipeline mirrors the Jupyter notebook version exactly.
        """
        h, w = frame.shape[:2]
        debug_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # --- Light mask ---
        mask_lights = detector._create_red_mask(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_CLOSE, kernel)
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_OPEN, kernel)
        debug_frame[mask_lights > 0] = (0, 0, 255)

        # --- Plate contour pipeline ---
        if features_dict and 'outer' in features_dict:
            try:
                outer = features_dict['outer']
                bottom_points = features_dict.get('bottom', outer)

                fari_bottom_y = int(max(bottom_points[0][1], bottom_points[1][1]))
                scale = abs(outer[1][0] - outer[0][0])   # horizontal span of lights

                # ROI x-range: same margins as the detector (±10%)
                x1 = int(min(outer[0][0], outer[1][0]))
                x2 = int(max(outer[0][0], outer[1][0]))
                margin_x = int(scale * 0.10)
                x1 = max(0, x1 - margin_x)
                x2 = min(w - 1, x2 + margin_x)

                # ROI y-range: 15%–45% below the bottom of the lights
                y1 = max(0, int(fari_bottom_y + 0.15 * scale))
                y2 = min(h - 1, int(fari_bottom_y + 0.45 * scale))

                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(debug_frame, "PLATE ROI", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    raise ValueError("Empty ROI")

                # Brightness mask inside ROI
                V_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
                mask_plate = cv2.inRange(V_roi, detector.v_plate_low, detector.v_plate_high)
                mask_plate = cv2.morphologyEx(mask_plate,
                                              cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))

                contours_plate, _ = cv2.findContours(mask_plate,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
                if not contours_plate:
                    raise ValueError("No plate contour found")

                largest = max(contours_plate, key=cv2.contourArea)
                mask_cluster = np.zeros_like(mask_plate)
                cv2.drawContours(mask_cluster, [largest], -1, 255, -1)

                ys, xs = np.where(mask_cluster > 0)
                if len(xs) == 0:
                    raise ValueError("Empty cluster mask")

                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                pad_x = int(0.25 * (x_max - x_min))
                pad_y = int(0.40 * (y_max - y_min))
                roi_plate = roi[
                    max(0, y_min - pad_y):min(roi.shape[0], y_max + pad_y),
                    max(0, x_min - pad_x):min(roi.shape[1], x_max + pad_x)
                ]
                roi_x_min = max(0, x_min - pad_x)
                roi_y_min = max(0, y_min - pad_y)

                if roi_plate.size == 0:
                    raise ValueError("Empty plate ROI")

                # Gradient-based contour extraction
                gray = cv2.cvtColor(roi_plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

                V_plate = cv2.cvtColor(roi_plate, cv2.COLOR_BGR2HSV)[:, :, 2]
                mask_light = cv2.inRange(V_plate, detector.v_plate_low, detector.v_plate_high)
                mask_light_expanded = cv2.dilate(
                    mask_light,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15)),
                    iterations=2
                )

                _, grad_binary = cv2.threshold(gradient, 0, 255,
                                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                grad_filtered = cv2.bitwise_and(grad_binary, mask_light_expanded)

                # Close horizontally and vertically, then OR the results
                grad_h = cv2.morphologyEx(grad_filtered, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
                grad_v = cv2.morphologyEx(grad_filtered, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
                grad_closed = cv2.bitwise_or(grad_h, grad_v)

                contours_grad, _ = cv2.findContours(grad_closed,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                if contours_grad:
                    main_contour = max(contours_grad, key=cv2.contourArea).copy()
                    main_contour[:, 0, 0] += roi_x_min + x1
                    main_contour[:, 0, 1] += roi_y_min + y1
                    cv2.drawContours(debug_frame, [main_contour], -1, (0, 255, 0), 2)

            except Exception as e:
                print(f"[create_debug_mask_frame] plate pipeline error: {e}")

        # --- Project 3D origin (rear axle center) ---
        origin_2d = None
        if rvec is not None and tvec is not None and camera_matrix is not None:
            try:
                projected, _ = cv2.projectPoints(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    rvec, tvec, camera_matrix,
                    dist_coeffs if dist_coeffs is not None else np.zeros(5)
                )
                origin_2d = tuple(map(int, projected[0][0]))
            except Exception as e:
                print(f"[create_debug_mask_frame] origin projection error: {e}")

        # --- Light anchor points ---
        if features_dict:
            point_styles = {
                'top':    (8,  (255, 255, 0)),   # cyan
                'outer':  (10, (0,   255, 0)),   # green (PnP reference)
                'bottom': (8,  (0,   255, 255)), # yellow
            }
            for key, (radius, color) in point_styles.items():
                for pt in features_dict.get(key, []):
                    cv2.circle(debug_frame, tuple(map(int, pt)), radius, color, -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), radius + 2, (255, 255, 255), 2)

        # --- Plate bottom edge ---
        bl, br = None, None
        if detector.prev_plate_corners is not None:
            bl = detector.prev_plate_corners['BL']
            br = detector.prev_plate_corners['BR']
        elif plate_bottom is not None and len(plate_bottom) == 2:
            bl = tuple(map(int, plate_bottom[0]))
            br = tuple(map(int, plate_bottom[1]))

        if bl is not None and br is not None:
            for pt in [bl, br]:
                cv2.circle(debug_frame, pt, 9,  (255, 0, 255), -1)
                cv2.circle(debug_frame, pt, 11, (255, 255, 255), 2)
            cv2.line(debug_frame, bl, br, (255, 0, 255), 3, cv2.LINE_AA)
            mid = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)
            cv2.putText(debug_frame, "PLATE", (mid[0] - 30, mid[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # --- 3D origin marker ---
        if origin_2d is not None:
            ox, oy = origin_2d
            s = 25
            orange = (0, 165, 255)
            cv2.line(debug_frame, (ox - s, oy), (ox + s, oy), orange, 4, cv2.LINE_AA)
            cv2.line(debug_frame, (ox, oy - s), (ox, oy + s), orange, 4, cv2.LINE_AA)
            cv2.circle(debug_frame, origin_2d, 18, orange, 3)
            cv2.circle(debug_frame, origin_2d, 25, (255, 255, 255), 2)

            label = "ORIGIN 3D"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (tw, text_h), _ = cv2.getTextSize(label, font, fs, th)
            lx, ly = ox + 30, oy - 15
            cv2.rectangle(debug_frame, (lx - 3, ly - text_h - 3), (lx + tw + 3, ly + 3),
                          (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (lx, ly), font, fs, orange, th, cv2.LINE_AA)

        # --- Header and legend ---
        cv2.putText(debug_frame, "DEBUG MASK VIEW", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Frame: {frame_idx}", (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        legend = [
            ("Red: light mask",        (0,   0,   255)),
            ("Green: plate contour",   (0,   255, 0)),
            ("Cyan: top lights",       (255, 255, 0)),
            ("Green: outer (ref)",     (0,   255, 0)),
            ("Yellow: bottom lights",  (0,   255, 255)),
            ("Magenta: plate bottom",  (255, 0,   255)),
            ("Orange: 3D origin",      (0,   165, 255)),
        ]
        for i, (label, color) in enumerate(legend):
            y = 60 + i * 25
            cv2.circle(debug_frame, (20, y), 6, color, -1)
            cv2.putText(debug_frame, label, (35, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return debug_frame


# ---------------------------------------------------------------------------
# Backward-compatibility wrappers
# ---------------------------------------------------------------------------

def draw_vanishing_points_complete(
    frame: np.ndarray,
    lights_frame1: Optional[np.ndarray],
    lights_frame2: Optional[np.ndarray],
    Vx: Optional[np.ndarray],
    Vy: Optional[np.ndarray],
    dot_product: Optional[float] = None,
    show_lines: bool = True,
    show_labels: bool = True
) -> None:
    """Deprecated wrapper kept for backward compatibility. Use DrawUtils.draw_vp_convergence."""
    if any(v is None for v in [lights_frame1, lights_frame2, Vx, Vy]):
        return
    DrawUtils.draw_vp_convergence(
        frame,
        features_curr={'outer': lights_frame1},
        features_prev={'outer': lights_frame2},
        vx_motion=Vx,
        vy_lateral=Vy,
        show_labels=show_labels
    )


def draw_plate_roi(frame, roi):
    """Draws a magenta rectangle around the license plate ROI."""
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "PLATE ROI", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)