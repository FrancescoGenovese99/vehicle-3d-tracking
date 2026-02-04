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
    elif motion_type == "ROTATION":
        bg_color = (0, 0, 128)
    else:
        bg_color = (0, 82, 128)
    
    text = f"MOTION: {motion_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (w - text_w) // 2
    y = 50
    
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
    def extend_line_bidirectional(frame, p1, p2, vp, color, thickness=2, gap=8, length_multiplier=2.5):
        """
        GEOMETRIA CORRETTA: Prolunga in ENTRAMBE le direzioni.
        
        - Linea SOLIDA: p1 -> p2
        - Linea TRATTEGGIATA: p2 -> avanti (direzione p1→p2)
        - Linea TRATTEGGIATA: p1 -> indietro (direzione opposta)
        
        Args:
            p1, p2: Punti del segmento originale
            vp: Vanishing point (non usato, ma mantenuto per compatibilità)
            color: Colore
            thickness: Spessore
            gap: Gap linea tratteggiata
            length_multiplier: Quanto estendere
        """
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        
        # ===== LINEA SOLIDA p1 -> p2 =====
        cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), 
                color, thickness, cv2.LINE_AA)
        
        # ===== DIREZIONE: da p1 verso p2 =====
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 1:
            return
        
        direction = direction / length  # Normalizza
        
        # ===== ESTENSIONE AVANTI: da p2 nella direzione p1→p2 =====
        extension_length = length * length_multiplier
        end_point_forward = p2 + direction * extension_length
        
        draw_dashed_line(frame, tuple(map(int, p2)), tuple(map(int, end_point_forward)),
                        color, thickness, gap)
        
        # ===== ESTENSIONE INDIETRO: da p1 nella direzione opposta =====
        end_point_backward = p1 - direction * extension_length
        
        draw_dashed_line(frame, tuple(map(int, p1)), tuple(map(int, end_point_backward)),
                        color, thickness, gap)
    
    @staticmethod
    def draw_vanishing_points_multifeature(
        frame: np.ndarray,
        features_t1: dict,
        features_t2: dict,
        plate_bottom_t1: Optional[np.ndarray],
        plate_bottom_t2: Optional[np.ndarray],
        Vx: np.ndarray,
        Vy: np.ndarray,
        show_lines: bool = True,
        show_labels: bool = True
    ):
        """
        Disegna vanishing points con GEOMETRIA CORRETTA.
        
        GEOMETRIA:
        - Segmenti orizzontali (L-R) prolungati → convergono a Vx
        - Traiettorie verticali (L1-L2, R1-R2) prolungate → convergono a Vy
        """
        if Vx is None or Vy is None:
            return
        
        h, w = frame.shape[:2]
        Vx_int = tuple(map(int, Vx))
        Vy_int = tuple(map(int, Vy))
        
        # Colori per features
        colors = {
            'top': (255, 255, 0),      # Ciano
            'outer': (0, 255, 0),      # Verde
            'plate_bottom': (255, 0, 255)  # Magenta
        }
        
        # Colori per traiettorie
        trajectory_colors = {
            'top': (128, 128, 255),     # Blu chiaro
            'outer': (255, 0, 0),       # Blu scuro
        }
        
        if show_lines:
            # ===== Vx: SEGMENTI ORIZZONTALI (L-R nello stesso frame) =====
            for feature_name in ['top', 'outer']:
                if feature_name not in features_t1 or feature_name not in features_t2:
                    continue
                
                color = colors[feature_name]
                pts1 = features_t1[feature_name]
                pts2 = features_t2[feature_name]
                
                L1, R1 = pts1[0], pts1[1]
                L2, R2 = pts2[0], pts2[1]
                
                # Frame t: L1-R1 bidirezionale
                DrawUtils.extend_line_bidirectional(frame, L1, R1, Vx_int, color, 1, gap=8, length_multiplier=2.5)
                
                # Frame t+1: L2-R2 bidirezionale (più spesso)
                DrawUtils.extend_line_bidirectional(frame, L2, R2, Vx_int, color, 2, gap=8, length_multiplier=2.5)
            
            # ===== PLATE BOTTOM → Vx =====
            if plate_bottom_t1 is not None and plate_bottom_t2 is not None:
                color = colors['plate_bottom']
                
                BL1, BR1 = plate_bottom_t1[0], plate_bottom_t1[1]
                BL2, BR2 = plate_bottom_t2[0], plate_bottom_t2[1]
                
                # Disegna punti
                for pt in [BL1, BR1]:
                    cv2.circle(frame, tuple(map(int, pt)), 4, color, -1)
                
                for pt in [BL2, BR2]:
                    cv2.circle(frame, tuple(map(int, pt)), 6, color, -1)
                    cv2.circle(frame, tuple(map(int, pt)), 8, (255, 255, 255), 2)
                
                # Segmenti bidirezionali
                DrawUtils.extend_line_bidirectional(frame, BL1, BR1, Vx_int, color, 2, gap=8, length_multiplier=2.0)
                DrawUtils.extend_line_bidirectional(frame, BL2, BR2, Vx_int, color, 3, gap=8, length_multiplier=2.0)
                
                # Label
                mid_x = int((BL2[0] + BR2[0]) / 2)
                mid_y = int((BL2[1] + BR2[1]) / 2)
                cv2.putText(frame, "PLATE", (mid_x - 30, mid_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # ===== Vy: TRAIETTORIE (L1→L2, R1→R2 tra frames) =====
            for feature_name in ['top', 'outer']:
                if feature_name not in features_t1 or feature_name not in features_t2:
                    continue
                
                pts1 = features_t1[feature_name]
                pts2 = features_t2[feature_name]
                
                L1, R1 = pts1[0], pts1[1]
                L2, R2 = pts2[0], pts2[1]
                
                traj_color = trajectory_colors[feature_name]
                
                # Traiettoria LEFT: L1→L2 bidirezionale
                DrawUtils.extend_line_bidirectional(frame, L1, L2, Vy_int, traj_color, 2, gap=8, length_multiplier=2.0)
                
                # Traiettoria RIGHT: R1→R2 bidirezionale
                DrawUtils.extend_line_bidirectional(frame, R1, R2, Vy_int, traj_color, 2, gap=8, length_multiplier=2.0)
        
        # ===== VANISHING POINTS =====
        # Vx
        if 0 <= Vx_int[0] < w and 0 <= Vx_int[1] < h:
            cv2.circle(frame, Vx_int, 12, (0, 255, 0), -1)
            cv2.circle(frame, Vx_int, 15, (255, 255, 255), 3)
            if show_labels:
                cv2.putText(frame, "Vx (lateral)", (Vx_int[0] + 20, Vx_int[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Vx @ ({Vx_int[0]}, {Vx_int[1]})", (10, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Vy
        if 0 <= Vy_int[0] < w and 0 <= Vy_int[1] < h:
            cv2.circle(frame, Vy_int, 12, (255, 0, 0), -1)
            cv2.circle(frame, Vy_int, 15, (255, 255, 255), 3)
            if show_labels:
                cv2.putText(frame, "Vy (motion)", (Vy_int[0] + 20, Vy_int[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, f"Vy @ ({Vy_int[0]}, {Vy_int[1]})", (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # ===== LEGENDA (UNA SOLA VOLTA!) =====
        if show_labels:
            y_offset = 100
            legend_items = [
                ('Lateral segments (->Vx):', None),
                ('  top L-R', colors['top']),
                ('  outer L-R', colors['outer']),
                ('  plate bottom', colors['plate_bottom']),
                ('', None),
                ('Motion trajectories (->Vy):', None),
                ('  top L/R', trajectory_colors['top']),
                ('  outer L/R', trajectory_colors['outer']),
            ]
            
            for name, color in legend_items:
                if color is None:
                    cv2.putText(frame, name, (10, y_offset + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (10, y_offset), (30, y_offset + 15), color, -1)
                    cv2.putText(frame, name, (35, y_offset + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
    
    @staticmethod
    def create_debug_mask_frame(
        frame: np.ndarray,
        detector,
        features_dict: Optional[dict],
        plate_bottom: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Debug mask: mostra processing LIVE della targa.
        
        MOSTRA:
        - Maschera fari (rosso)
        - ROI targa (giallo)
        - Contorno processato targa (verde brillante)
        - Punti identificati
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
        
        # ===== PUNTI IDENTIFICATI =====
        if features_dict:
            if 'top' in features_dict:
                for pt in features_dict['top']:
                    cv2.circle(debug_frame, tuple(map(int, pt)), 8, (255, 255, 0), -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), 10, (255, 255, 255), 2)
            
            if 'outer' in features_dict:
                for pt in features_dict['outer']:
                    cv2.circle(debug_frame, tuple(map(int, pt)), 10, (0, 255, 0), -1)
                    cv2.circle(debug_frame, tuple(map(int, pt)), 12, (255, 255, 255), 2)
        
        # Plate bottom
        if plate_bottom is not None:
            for pt in plate_bottom:
                cv2.circle(debug_frame, tuple(map(int, pt)), 12, (255, 0, 255), -1)
                cv2.circle(debug_frame, tuple(map(int, pt)), 14, (255, 255, 255), 2)
            
            BL, BR = plate_bottom[0], plate_bottom[1]
            cv2.line(debug_frame, tuple(map(int, BL)), tuple(map(int, BR)), 
                    (255, 0, 255), 3, cv2.LINE_AA)
        
        # ===== LEGENDA =====
        cv2.putText(debug_frame, "DEBUG MASK VIEW", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        legend_y = 60
        legend_items = [
            ("Red: Light mask", (0, 0, 255)),
            ("Green: Plate contour", (0, 255, 0)),
            ("Cyan: Top", (255, 255, 0)),
            ("Green: Outer (ref)", (0, 255, 0)),
            ("Yellow: Bottom", (0, 255, 255)),
            ("Magenta: Plate Bottom", (255, 0, 255))
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

