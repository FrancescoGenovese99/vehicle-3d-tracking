"""
Draw Utils - VISUALIZZAZIONE CORRETTA COMPLETA - FIXED GEOMETRIA
FIX CRITICI:
1. extend_line_bidirectional: prolunga in ENTRAMBE le direzioni
2. Debug mask: mostra CONTORNO PROCESSATO come nello script Jupiter
3. Visualizzazione VP: linee convergono CORRETTAMENTE ai vanishing points
4. NESSUNA duplicazione legenda
5. Indentazione CORRETTA
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


def draw_tracking_info(frame: np.ndarray,
                      frame_idx: int,
                      method: str,
                      distance: float,
                      num_points: int) -> None:
    """Disegna info tracking in alto a sinistra."""
    y_offset = 30
    cv2.putText(frame, f"Frame: {frame_idx}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Method: {method}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Distance: {distance:.2f}m", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Points: {num_points}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_motion_type_overlay(frame: np.ndarray,
                             motion_type: str,
                             bg_alpha: float = 0.6) -> None:
    """Disegna overlay tipo movimento."""
    h, w = frame.shape[:2]
    
    text_color = (0, 255, 255)
    
    if motion_type == "TRANSLATION":
        bg_color = (0, 128, 0)
    elif motion_type == "STEERING":
        bg_color = (0, 0, 128)
    else:
        bg_color = (0, 82, 128)
    
    text = f"MOTION: {motion_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (w - text_w) // 2
    y = h - 20  # spostato in basso, non copre tracking info in alto
    
    overlay = frame.copy()
    
    padding = 15
    cv2.rectangle(overlay,
                 (x - padding, y - text_h - padding),
                 (x + text_w + padding, y + baseline + padding),
                 bg_color, -1)
    
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    cv2.putText(frame, text, (x, y),
               font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_3d_axes(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float = 1.5,
    thickness: int = 4
) -> None:
    """Disegna assi 3D del sistema veicolo."""
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)
    
    projected, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    projected = projected.reshape(-1, 2).astype(int)
    
    origin = tuple(projected[0])
    x_end = tuple(projected[1])
    y_end = tuple(projected[2])
    z_end = tuple(projected[3])
    
    cv2.line(frame, origin, x_end, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.line(frame, origin, y_end, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(frame, origin, z_end, (255, 0, 0), thickness, cv2.LINE_AA)
    
    cv2.circle(frame, origin, 8, (255, 255, 255), -1)
    cv2.circle(frame, origin, 10, (0, 0, 0), 2)
    
    def draw_label_with_bg(img, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(img, 
                     (pos[0] - 2, pos[1] - th - 2),
                     (pos[0] + tw + 2, pos[1] + 2),
                     (0, 0, 0), -1)
        
        cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    
    draw_label_with_bg(frame, 'X (forward)', x_end, (0, 0, 255))
    draw_label_with_bg(frame, 'Y (left)', y_end, (0, 255, 0))
    draw_label_with_bg(frame, 'Z (up)', z_end, (255, 0, 0))
    draw_label_with_bg(frame, 'Origin', (origin[0] + 15, origin[1] - 10), (255, 255, 255))


def draw_bbox_3d(frame: np.ndarray,
                projected_points: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2) -> None:
    """Disegna bounding box 3D."""
    if projected_points is None or len(projected_points) != 8:
        return
    
    pts = projected_points.astype(int)
    
    # Base
    base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in base_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Top
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    for i, j in top_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Verticali
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in vertical_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Posteriore evidenziato
    rear_color = (0, 0, 255)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), rear_color, thickness + 2)
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), rear_color, thickness + 2)


def draw_dashed_line(frame, pt1, pt2, color, thickness=1, gap=10):
    """Disegna linea tratteggiata."""
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    if dist < 1:
        return
    
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        pts.append((x, y))
    
    for i in range(0, len(pts) - 1, 2):
        if i + 1 < len(pts):
            cv2.line(frame, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)


class DrawUtils:
    """Classe helper per visualizzazione - GEOMETRIA CORRETTA."""
    
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
        Disegna vanishing points con linee convergenti.
        
        Vy (laterale): linee tra punti L-R → Vy (CIANO)
        Vx (movimento): traiettorie t→t+1 → Vx (ARANCIONE)
        
        Args:
            frame: Frame su cui disegnare
            features_curr: Features frame corrente {'outer': [...], 'top': [...], ...}
            features_prev: Features frame precedente (per Vx motion)
            vx_motion: VP movimento (array [x, y] o None)
            vy_lateral: VP laterale (array [x, y] o None)
            show_labels: Mostra etichette VP
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        def clamp_vp(vp):
            """Clamp VP a range ragionevole."""
            if vp is None:
                return None
            x = np.clip(vp[0], -w*2, w*3)
            y = np.clip(vp[1], -h*2, h*3)
            return (int(x), int(y))
        
        def is_on_screen(pt, margin=100):
            """Check se punto è visibile (con margine)."""
            return -margin <= pt[0] <= w+margin and -margin <= pt[1] <= h+margin
        
        # ===== Vy LATERALE (CIANO) =====
        if vy_lateral is not None:
            vy_pt = clamp_vp(vy_lateral)
            vy_on_screen = is_on_screen(vy_pt, margin=200)
            
            color_vy = (255, 220, 0)  # Ciano
            
            # Linee da coppie L-R → Vy
            pairs = []
            if 'outer' in features_curr:
                o = features_curr['outer']
                if len(o) == 2:
                    pairs.append((o[0], o[1], 'outer'))
            
            if 'top' in features_curr:
                t = features_curr['top']
                if len(t) == 2:
                    pairs.append((t[0], t[1], 'top'))
            
            if 'plate_bottom' in features_curr:
                pb = features_curr['plate_bottom']
                if len(pb) == 2:
                    pairs.append((pb[0], pb[1], 'plate'))
            
            for pL, pR, name in pairs:
                pL_int = (int(pL[0]), int(pL[1]))
                pR_int = (int(pR[0]), int(pR[1]))
                
                # Linea solida tra L e R
                cv2.line(overlay, pL_int, pR_int, color_vy, 2, cv2.LINE_AA)
                
                # Linee tratteggiate → Vy
                if vy_on_screen:
                    draw_dashed_line(overlay, pL_int, vy_pt, color_vy, 1, gap=8)
                    draw_dashed_line(overlay, pR_int, vy_pt, color_vy, 1, gap=8)
            
            # Disegna Vy point
            if vy_on_screen:
                cv2.circle(overlay, vy_pt, 12, color_vy, -1)
                cv2.circle(overlay, vy_pt, 15, (255, 255, 255), 2)
                if show_labels:
                    cv2.putText(overlay, "Vy (lateral)", 
                               (vy_pt[0] + 20, vy_pt[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vy, 2, cv2.LINE_AA)
            else:
                # Fuori schermo: mostra freccia
                center = np.array([w/2, h/2])
                direction = np.array(vy_pt, dtype=float) - center
                norm = np.linalg.norm(direction)
                if norm > 1:
                    direction /= norm
                    arrow_end = (center + direction * 80).astype(int)
                    cv2.arrowedLine(overlay, tuple(center.astype(int)),
                                   tuple(arrow_end), color_vy, 3, tipLength=0.3)
                cv2.putText(overlay, f"Vy→({int(vy_lateral[0])},{int(vy_lateral[1])})",
                           (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vy, 2)
        
        # ===== Vx MOVIMENTO (ARANCIONE) =====
        if vx_motion is not None and features_prev is not None:
            vx_pt = clamp_vp(vx_motion)
            vx_on_screen = is_on_screen(vx_pt, margin=200)
            
            color_vx = (0, 140, 255)  # Arancione
            
            # Linee da traiettorie t→t+1 → Vx
            trajectories = []
            
            for key in ['outer', 'top']:
                if key in features_curr and key in features_prev:
                    curr = features_curr[key]
                    prev = features_prev[key]
                    
                    for i in range(min(len(curr), len(prev))):
                        p_prev = (int(prev[i][0]), int(prev[i][1]))
                        p_curr = (int(curr[i][0]), int(curr[i][1]))
                        
                        # Check movimento minimo
                        movement = np.linalg.norm(np.array(p_curr) - np.array(p_prev))
                        if movement < 1.0:
                            continue
                        
                        trajectories.append((p_prev, p_curr, key, i))
            
            # Disegna traiettorie
            for p_prev, p_curr, key, idx in trajectories:
                # Freccia da prev → curr
                cv2.arrowedLine(overlay, p_prev, p_curr, color_vx, 2, 
                               tipLength=0.2, line_type=cv2.LINE_AA)
                
                # Linea tratteggiata → Vx
                if vx_on_screen:
                    draw_dashed_line(overlay, p_curr, vx_pt, color_vx, 1, gap=8)
            
            # Disegna Vx point
            if vx_on_screen:
                cv2.circle(overlay, vx_pt, 12, color_vx, -1)
                cv2.circle(overlay, vx_pt, 15, (255, 255, 255), 2)
                if show_labels:
                    cv2.putText(overlay, "Vx (motion)", 
                               (vx_pt[0] + 20, vx_pt[1] + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vx, 2, cv2.LINE_AA)
            else:
                # Fuori schermo: mostra freccia
                center = np.array([w/2, h/2])
                direction = np.array(vx_pt, dtype=float) - center
                norm = np.linalg.norm(direction)
                if norm > 1:
                    direction /= norm
                    arrow_end = (center + direction * 80).astype(int)
                    cv2.arrowedLine(overlay, tuple(center.astype(int)),
                                   tuple(arrow_end), color_vx, 3, tipLength=0.3)
                cv2.putText(overlay, f"Vx→({int(vx_motion[0])},{int(vx_motion[1])})",
                           (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vx, 2)
        
        # Blend semitrasparente
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
        frame_idx: int = 0  # ← NUOVO PARAMETRO
    ) -> np.ndarray:
        """
        Debug mask: mostra processing LIVE della targa.
        
        MOSTRA:
        - Maschera fari (rosso)
        - ROI targa (giallo)
        - Contorno processato targa (verde brillante)
        - Punti identificati (top, outer, bottom)
        - Plate bottom (BL, BR) - SOLO bordo inferiore
        - Origine sistema riferimento 3D (tra gomme posteriori a terra)
        """
        h, w = frame.shape[:2]
        debug_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # ===== MASCHERA FARI =====
        mask_lights = detector._create_red_mask(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_CLOSE, kernel)
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_OPEN, kernel)
        
        debug_frame[mask_lights > 0] = (0, 0, 255)  # Rosso
        
        # ===== PROCESSING TARGA LIVE =====
        if features_dict and 'outer' in features_dict:
            try:
                from scipy import stats
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                V = hsv[:, :, 2]
                
                outer_points = features_dict['outer']
                
                # Trova Y base dai fari
                outer = features_dict['outer']

                # ✅ FIX: USA BOTTOM, NON OUTER!
                bottom_points = features_dict.get('bottom')
                if bottom_points is None:
                    bottom_points = outer  # Fallback
                
                fari_bottom_y = int(max(bottom_points[0][1], bottom_points[1][1]))
                scale = abs(outer[1][0] - outer[0][0])  # lights width
                
                # ✅ FIX: ROI COERENTE CON DETECTOR
                x1 = int(min(outer[0][0], outer[1][0]))
                x2 = int(max(outer[0][0], outer[1][0]))
                
                margin_x = int(scale * 0.10)  # 10% margine
                x1 = max(0, x1 - margin_x)
                x2 = min(w - 1, x2 + margin_x)
                
                # ✅ FIX: RANGE VERTICALE CORRETTO (0.15-0.45)
                y1 = int(fari_bottom_y + 0.15 * scale)
                y2 = int(fari_bottom_y + 0.45 * scale)
                
                # Clamp
                y1 = max(0, y1)
                y2 = min(h - 1, y2)

                # Disegna ROI box
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    debug_frame,
                    "ROI TARGA",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    # Processing completo come Jupiter
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    V_roi = hsv_roi[:, :, 2]

                    mask_plate = cv2.inRange(V_roi, detector.v_plate_low, detector.v_plate_high)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
                    mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)

                    contours_plate, _ = cv2.findContours(
                        mask_plate,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours_plate:
                        largest = max(contours_plate, key=cv2.contourArea)
                        mask_cluster = np.zeros_like(mask_plate)
                        cv2.drawContours(mask_cluster, [largest], -1, 255, -1)

                        ys, xs = np.where(mask_cluster > 0)
                        if len(xs) > 0:
                            x_min, x_max = xs.min(), xs.max()
                            y_min, y_max = ys.min(), ys.max()

                            pad_x = int(0.25 * (x_max - x_min))
                            pad_y = int(0.40 * (y_max - y_min))

                            roi_x_min = max(0, x_min - pad_x)
                            roi_x_max = min(roi.shape[1], x_max + pad_x)
                            roi_y_min = max(0, y_min - pad_y)
                            roi_y_max = min(roi.shape[0], y_max + pad_y)

                            roi_plate = roi[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

                            if roi_plate.size > 0:
                                # Gradient
                                gray_plate = cv2.cvtColor(roi_plate, cv2.COLOR_BGR2GRAY)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                gray_plate = clahe.apply(gray_plate)
                                gray_plate = cv2.GaussianBlur(gray_plate, (7, 7), 0)

                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                gradient = cv2.morphologyEx(gray_plate, cv2.MORPH_GRADIENT, kernel)

                                hsv_plate = cv2.cvtColor(roi_plate, cv2.COLOR_BGR2HSV)
                                V_plate = hsv_plate[:, :, 2]
                                mask_light = cv2.inRange(
                                    V_plate,
                                    detector.v_plate_low,
                                    detector.v_plate_high
                                )

                                kernel_expand = cv2.getStructuringElement(
                                    cv2.MORPH_RECT, (20, 15)
                                )
                                mask_light_expanded = cv2.dilate(
                                    mask_light, kernel_expand, iterations=2
                                )

                                _, gradient_binary = cv2.threshold(
                                    gradient, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                )

                                gradient_filtered = cv2.bitwise_and(
                                    gradient_binary, mask_light_expanded
                                )

                                kernel_close_h = cv2.getStructuringElement(
                                    cv2.MORPH_RECT, (2, 1)
                                )
                                kernel_close_v = cv2.getStructuringElement(
                                    cv2.MORPH_RECT, (1, 2)
                                )

                                gradient_closed_h = cv2.morphologyEx(
                                    gradient_filtered,
                                    cv2.MORPH_CLOSE,
                                    kernel_close_h
                                )
                                gradient_closed_v = cv2.morphologyEx(
                                    gradient_filtered,
                                    cv2.MORPH_CLOSE,
                                    kernel_close_v
                                )

                                gradient_closed = cv2.bitwise_or(
                                    gradient_closed_h,
                                    gradient_closed_v
                                )

                                # CONTORNO FINALE
                                contours_grad, _ = cv2.findContours(
                                    gradient_closed,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE
                                )

                                if contours_grad:
                                    main_contour = max(
                                        contours_grad, key=cv2.contourArea
                                    )

                                    # Coordinate globali
                                    global_contour = main_contour.copy()
                                    global_contour[:, 0, 0] += roi_x_min + x1
                                    global_contour[:, 0, 1] += roi_y_min + y1

                                    # VERDE BRILLANTE
                                    cv2.drawContours(
                                        debug_frame,
                                        [global_contour],
                                        -1,
                                        (0, 255, 0),
                                        2
                                    )

            
            except Exception as e:
                print(f"Debug mask error: {e}")
        
        # ===== CALCOLA ORIGINE 3D (TRA GOMME POSTERIORI) =====
        origin_3d_projected = None
        if rvec is not None and tvec is not None and camera_matrix is not None:
            try:
                # Punto origine del sistema di riferimento veicolo [0, 0, 0]
                # (tra le gomme posteriori a terra)
                origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                
                # Proietta sul piano immagine
                projected, _ = cv2.projectPoints(
                    origin_3d, rvec, tvec, camera_matrix, 
                    dist_coeffs if dist_coeffs is not None else np.zeros(5)
                )
                
                origin_3d_projected = tuple(map(int, projected[0][0]))
                
            except Exception as e:
                print(f"Origin projection error: {e}")
        
        # ===== PUNTI IDENTIFICATI (FARI) =====
        if features_dict:
            # Top = ciano
            if 'top' in features_dict:
                for pt in features_dict['top']:
                    cv2.circle(debug_frame, tuple(map(int, pt)), 8, (255, 255, 0), -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), 10, (255, 255, 255), 2)
            
            # Outer = verde brillante (reference points)
            if 'outer' in features_dict:
                for pt in features_dict['outer']:
                    cv2.circle(debug_frame, tuple(map(int, pt)), 10, (0, 255, 0), -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), 12, (255, 255, 255), 2)
            
            # Bottom = giallo
            if 'bottom' in features_dict:
                for pt in features_dict['bottom']:
                    cv2.circle(debug_frame, tuple(map(int, pt)), 8, (0, 255, 255), -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), 10, (255, 255, 255), 2)
        
        # ===== PLATE BOTTOM (SOLO BL, BR) =====
        if detector.prev_plate_corners is not None:
            plate_corners = detector.prev_plate_corners
            BL = plate_corners['BL']
            BR = plate_corners['BR']
            
            # SOLO angoli BOTTOM (BL, BR) = magenta
            for pt in [BL, BR]:
                cv2.circle(debug_frame, pt, 9, (255, 0, 255), -1)
                cv2.circle(debug_frame, pt, 11, (255, 255, 255), 2)
            
            # SOLO linea bordo inferiore (spessa)
            cv2.line(debug_frame, BL, BR, (255, 0, 255), 3, cv2.LINE_AA)
            
            # Label
            mid_x = int((BL[0] + BR[0]) / 2)
            mid_y = int((BL[1] + BR[1]) / 2)
            cv2.putText(debug_frame, "PLATE", (mid_x - 30, mid_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Fallback: usa plate_bottom se prev_plate_corners non disponibile
        elif plate_bottom is not None:
            for pt in plate_bottom:
                cv2.circle(debug_frame, tuple(map(int, pt)), 9, (255, 0, 255), -1)
                cv2.circle(debug_frame, tuple(map(int, pt)), 11, (255, 255, 255), 2)
            
            BL, BR = plate_bottom[0], plate_bottom[1]
            cv2.line(debug_frame, tuple(map(int, BL)), tuple(map(int, BR)), 
                    (255, 0, 255), 3, cv2.LINE_AA)
        
        # ===== ORIGINE SISTEMA RIFERIMENTO 3D =====
        if origin_3d_projected is not None:
            # Croce arancione grande
            cross_size = 25
            cv2.line(debug_frame, 
                    (origin_3d_projected[0] - cross_size, origin_3d_projected[1]), 
                    (origin_3d_projected[0] + cross_size, origin_3d_projected[1]), 
                    (0, 165, 255), 4, cv2.LINE_AA)
            cv2.line(debug_frame, 
                    (origin_3d_projected[0], origin_3d_projected[1] - cross_size), 
                    (origin_3d_projected[0], origin_3d_projected[1] + cross_size), 
                    (0, 165, 255), 4, cv2.LINE_AA)
            
            # Cerchi concentrici
            cv2.circle(debug_frame, origin_3d_projected, 18, (0, 165, 255), 3)
            cv2.circle(debug_frame, origin_3d_projected, 25, (255, 255, 255), 2)
            
            # Label con sfondo
            label_text = "ORIGIN 3D"
            label_pos = (origin_3d_projected[0] + 30, origin_3d_projected[1] - 15)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Sfondo nero
            cv2.rectangle(debug_frame,
                        (label_pos[0] - 3, label_pos[1] - th - 3),
                        (label_pos[0] + tw + 3, label_pos[1] + 3),
                        (0, 0, 0), -1)
            
            cv2.putText(debug_frame, label_text, label_pos,
                    font, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        
        # ===== LEGENDA AGGIORNATA =====
        cv2.putText(debug_frame, "DEBUG MASK VIEW", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # ===== NUOVO: FRAME COUNT =====
        cv2.putText(debug_frame, f"Frame: {frame_idx}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        legend_y = 60
        legend_items = [
            ("Red: Light mask", (0, 0, 255)),
            ("Green: Plate contour", (0, 255, 0)),
            ("Cyan: Top lights", (255, 255, 0)),
            ("Green: Outer (ref)", (0, 255, 0)),
            ("Yellow: Bottom lights", (0, 255, 255)),
            ("Magenta: Plate bottom", (255, 0, 255)),
            ("Orange: Origin 3D", (0, 165, 255))
        ]
        
        for name, color in legend_items:
            cv2.circle(debug_frame, (20, legend_y), 6, color, -1)
            cv2.putText(debug_frame, name, (35, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        
        return debug_frame
    
# BACKWARD COMPATIBILITY
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
    """BACKWARD COMPATIBILITY wrapper."""
    if lights_frame1 is None or lights_frame2 is None or Vx is None or Vy is None:
        return
    
    features_t1 = {'outer': lights_frame1}
    features_t2 = {'outer': lights_frame2}
    
    DrawUtils.draw_vanishing_points_multifeature(
        frame, features_t1, features_t2, None, None, Vx, Vy, show_lines, show_labels
    )
    
def draw_plate_roi(frame, roi):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "PLATE ROI", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)