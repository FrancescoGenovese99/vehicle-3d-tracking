"""
Draw Utils - Funzioni di visualizzazione ESTESE per debug
Include: VP, assi 3D, piano π, info geometriche
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
    
    text_color = (0, 255, 255)  # Giallo
    
    if motion_type == "TRANSLATION":
        bg_color = (0, 128, 0)  # Verde scuro
    elif motion_type == "ROTATION":
        bg_color = (0, 0, 128)  # Rosso scuro
    else:
        bg_color = (0, 82, 128)  # Arancione scuro
    
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
    """
    Disegna COMPLETO dei vanishing points con tutte le geometrie.
    
    Args:
        frame: Frame da modificare in-place
        lights_frame1: Luci frame precedente [[L1_x,L1_y], [R1_x,R1_y]]
        lights_frame2: Luci frame corrente [[L2_x,L2_y], [R2_x,R2_y]]
        Vx: Vanishing point segmenti luci
        Vy: Vanishing point traiettorie
        dot_product: Prodotto scalare per perpendicolarità
        show_lines: Mostra linee geometriche
        show_labels: Mostra etichette
    """
    if lights_frame1 is None or lights_frame2 is None:
        return
    
    h, w = frame.shape[:2]
    
    L1, R1 = lights_frame1[0], lights_frame1[1]
    L2, R2 = lights_frame2[0], lights_frame2[1]
    
    # ========== LUCI CORRENTI ==========
    
    # Cerchi luci frame2 (più grandi, evidenziati)
    cv2.circle(frame, tuple(L2.astype(int)), 10, (255, 255, 0), -1)  # Cyan LEFT
    cv2.circle(frame, tuple(R2.astype(int)), 10, (255, 0, 255), -1)  # Magenta RIGHT
    
    # Bordo nero per contrasto
    cv2.circle(frame, tuple(L2.astype(int)), 12, (0, 0, 0), 2)
    cv2.circle(frame, tuple(R2.astype(int)), 12, (0, 0, 0), 2)
    
    if show_labels:
        cv2.putText(frame, "L2", (int(L2[0])-30, int(L2[1])-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "R2", (int(R2[0])+15, int(R2[1])-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Segmento luci frame2 (orizzontale, VERDE)
    cv2.line(frame, tuple(L2.astype(int)), tuple(R2.astype(int)),
            (0, 255, 0), 3)
    
    # ========== LUCI PRECEDENTI (più piccole, trasparenti) ==========
    
    if show_lines:
        # Cerchi luci frame1 (piccoli)
        cv2.circle(frame, tuple(L1.astype(int)), 5, (100, 255, 255), -1)
        cv2.circle(frame, tuple(R1.astype(int)), 5, (255, 100, 255), -1)
        
        if show_labels:
            cv2.putText(frame, "L1", (int(L1[0])-25, int(L1[1])+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
            cv2.putText(frame, "R1", (int(R1[0])+10, int(R1[1])+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
        
        # Segmento luci frame1 (orizzontale, VERDE chiaro)
        cv2.line(frame, tuple(L1.astype(int)), tuple(R1.astype(int)),
                (100, 255, 100), 2)
    
    # ========== VANISHING POINT Vx (SEGMENTI LUCI) ==========
    
    if Vx is not None and show_lines:
        # Linee dai segmenti verso Vx (GIALLO)
        # Da L1-R1 verso Vx
        cv2.line(frame, tuple(L1.astype(int)), tuple(Vx.astype(int)),
                (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, tuple(R1.astype(int)), tuple(Vx.astype(int)),
                (0, 255, 255), 2, cv2.LINE_AA)
        
        # Da L2-R2 verso Vx (più spesso)
        cv2.line(frame, tuple(L2.astype(int)), tuple(Vx.astype(int)),
                (0, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, tuple(R2.astype(int)), tuple(Vx.astype(int)),
                (0, 255, 255), 3, cv2.LINE_AA)
        
        # Disegna Vx
        if 0 <= Vx[0] < w and 0 <= Vx[1] < h:
            # Dentro frame: croce + cerchio
            cv2.drawMarker(frame, tuple(Vx.astype(int)), (0, 255, 255),
                         cv2.MARKER_CROSS, 40, 4)
            cv2.circle(frame, tuple(Vx.astype(int)), 15, (0, 255, 255), 3)
            cv2.circle(frame, tuple(Vx.astype(int)), 17, (0, 0, 0), 2)  # Bordo
            
            if show_labels:
                cv2.putText(frame, "Vx", (int(Vx[0])+25, int(Vx[1])-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(frame, "(light segments)", (int(Vx[0])+25, int(Vx[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Fuori frame: freccia direzionale
            center = np.array([w//2, h//2], dtype=float)
            direction = Vx - center
            direction = direction / np.linalg.norm(direction)
            
            edge_point = center + direction * min(w, h) * 0.4
            
            cv2.arrowedLine(frame, tuple(center.astype(int)), 
                          tuple(edge_point.astype(int)),
                          (0, 255, 255), 4, tipLength=0.3)
            
            if show_labels:
                cv2.putText(frame, "Vx →", tuple((edge_point + direction*30).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # ========== VANISHING POINT Vy (TRAIETTORIE) ==========
    
    if Vy is not None and show_lines:
        # Traiettorie (CYAN e MAGENTA)
        # L1 → L2 (traiettoria luce sinistra)
        cv2.line(frame, tuple(L1.astype(int)), tuple(L2.astype(int)),
                (255, 255, 0), 3)
        
        # R1 → R2 (traiettoria luce destra)
        cv2.line(frame, tuple(R1.astype(int)), tuple(R2.astype(int)),
                (255, 0, 255), 3)
        
        # Estensioni verso Vy
        if 0 <= Vy[0] < w*2 and 0 <= Vy[1] < h*2:
            cv2.line(frame, tuple(L2.astype(int)), tuple(Vy.astype(int)),
                    (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, tuple(R2.astype(int)), tuple(Vy.astype(int)),
                    (255, 0, 255), 2, cv2.LINE_AA)
        
        # Disegna Vy
        if 0 <= Vy[0] < w and 0 <= Vy[1] < h:
            # Dentro frame: croce + cerchio
            cv2.drawMarker(frame, tuple(Vy.astype(int)), (0, 0, 255),
                         cv2.MARKER_CROSS, 40, 4)
            cv2.circle(frame, tuple(Vy.astype(int)), 15, (0, 0, 255), 3)
            cv2.circle(frame, tuple(Vy.astype(int)), 17, (0, 0, 0), 2)  # Bordo
            
            if show_labels:
                cv2.putText(frame, "Vy", (int(Vy[0])+25, int(Vy[1])-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, "(trajectories)", (int(Vy[0])+25, int(Vy[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # Fuori frame: freccia direzionale
            center = np.array([w//2, h//2], dtype=float)
            direction = Vy - center
            direction = direction / np.linalg.norm(direction)
            
            edge_point = center + direction * min(w, h) * 0.35
            
            cv2.arrowedLine(frame, tuple(center.astype(int)), 
                          tuple(edge_point.astype(int)),
                          (0, 0, 255), 4, tipLength=0.3)
            
            if show_labels:
                cv2.putText(frame, "Vy →", tuple((edge_point + direction*30).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # ========== VANISHING LINE (se entrambi i VP sono visibili) ==========
    
    if Vx is not None and Vy is not None and show_lines:
        # Linea tratteggiata tra Vx e Vy (BIANCO)
        if (0 <= Vx[0] < w*2 and 0 <= Vx[1] < h*2 and
            0 <= Vy[0] < w*2 and 0 <= Vy[1] < h*2):
            
            # Disegna linea tratteggiata
            draw_dashed_line(frame, 
                           tuple(Vx.astype(int)), 
                           tuple(Vy.astype(int)),
                           (255, 255, 255), 2, gap=10)
            
            # Etichetta vanishing line
            if show_labels:
                mid = ((Vx + Vy) / 2).astype(int)
                if 0 <= mid[0] < w and 0 <= mid[1] < h:
                    cv2.putText(frame, "vanishing line l", tuple(mid),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ========== INFO PERPENDICOLARITÀ ==========
    
    if dot_product is not None:
        # Box info in basso a sinistra
        info_y = h - 80
        
        # Background semi-trasparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, info_y - 25), (400, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Testo
        perp_text = f"K^-1·Vx ⊥ K^-1·Vy: {dot_product:.4f}"
        color = (0, 255, 0) if dot_product < 0.2 else (0, 165, 255)
        
        cv2.putText(frame, perp_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        status = "PERPENDICULAR ✓" if dot_product < 0.2 else "NOT PERPENDICULAR ✗"
        cv2.putText(frame, status, (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_dashed_line(frame, pt1, pt2, color, thickness=1, gap=10):
    """Disegna linea tratteggiata."""
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        pts.append((x, y))
    
    for i in range(0, len(pts) - 1, 2):
        if i + 1 < len(pts):
            cv2.line(frame, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)


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
    Disegna assi 3D del sistema veicolo con etichette.
    
    Args:
        frame: Frame da modificare
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Matrice intrinseca K
        dist_coeffs: Coefficienti distorsione
        axis_length: Lunghezza assi in metri
        thickness: Spessore linee
    """
    # Definisci assi nel sistema veicolo
    axis_points = np.array([
        [0, 0, 0],                    # Origine (a terra, centro ruote post)
        [axis_length, 0, 0],          # X: avanti (ROSSO)
        [0, axis_length, 0],          # Y: sinistra (VERDE)
        [0, 0, axis_length]           # Z: su (BLU)
    ], dtype=np.float32)
    
    # Proietta assi
    projected, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    projected = projected.reshape(-1, 2).astype(int)
    
    origin = tuple(projected[0])
    x_end = tuple(projected[1])
    y_end = tuple(projected[2])
    z_end = tuple(projected[3])
    
    # Disegna assi con colori e spessore
    cv2.line(frame, origin, x_end, (0, 0, 255), thickness, cv2.LINE_AA)    # X: Rosso
    cv2.line(frame, origin, y_end, (0, 255, 0), thickness, cv2.LINE_AA)    # Y: Verde
    cv2.line(frame, origin, z_end, (255, 0, 0), thickness, cv2.LINE_AA)    # Z: Blu
    
    # Cerchio all'origine (BIANCO)
    cv2.circle(frame, origin, 8, (255, 255, 255), -1)
    cv2.circle(frame, origin, 10, (0, 0, 0), 2)
    
    # Etichette con background per leggibilità
    def draw_label_with_bg(img, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background nero
        cv2.rectangle(img, 
                     (pos[0] - 2, pos[1] - th - 2),
                     (pos[0] + tw + 2, pos[1] + 2),
                     (0, 0, 0), -1)
        
        # Testo
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
    
    # Posteriore evidenziato (ROSSO)
    rear_color = (0, 0, 255)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), rear_color, thickness + 2)
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), rear_color, thickness + 2)

def draw_bbox_with_validation(
    frame: np.ndarray,
    bbox_2d: np.ndarray,
    measured_lights: np.ndarray,
    theoretical_lights: Optional[np.ndarray],
    is_aligned: bool,
    alignment_error: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """
    Disegna bbox CON validazione visiva.
    
    Args:
        frame: Frame da modificare
        bbox_2d: Punti bbox proiettati (8, 2)
        measured_lights: Fari misurati [[L, R]]
        theoretical_lights: Fari teorici sulla bbox
        is_aligned: Se bbox è ben allineata
        alignment_error: Errore in pixel
        color: Colore bbox
        thickness: Spessore
    """
    if bbox_2d is None:
        return
    
    # Disegna bbox normale
    draw_bbox_3d(frame, bbox_2d, color, thickness)
    
    # ✅ CROCI VIOLA: dove DOVREBBERO stare i fari sulla bbox
    if theoretical_lights is not None:
        for i, light_pos in enumerate(theoretical_lights):
            x, y = int(light_pos[0]), int(light_pos[1])
            
            # Croce viola grande
            size = 15
            cv2.line(frame, (x - size, y), (x + size, y), (255, 0, 255), 3)
            cv2.line(frame, (x, y - size), (x, y + size), (255, 0, 255), 3)
            
            # Cerchio viola
            cv2.circle(frame, (x, y), 8, (255, 0, 255), 2)
            
            # Etichetta
            label = "TH-L" if i == 0 else "TH-R"
            cv2.putText(frame, label, (x + 20, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # ✅ LINEE DI CONNESSIONE: da fari misurati a teorici
    if theoretical_lights is not None and measured_lights is not None:
        for i in range(2):
            pt_meas = tuple(measured_lights[i].astype(int))
            pt_theo = tuple(theoretical_lights[i].astype(int))
            
            # Linea tratteggiata
            color_line = (0, 255, 0) if is_aligned else (0, 0, 255)
            draw_dashed_line(frame, pt_meas, pt_theo, color_line, 2, gap=5)
    
    # ✅ INFO ALLINEAMENTO
    status_text = f"Alignment: {'OK' if is_aligned else 'BAD'} ({alignment_error:.1f}px)"
    status_color = (0, 255, 0) if is_aligned else (0, 0, 255)
    
    cv2.putText(frame, status_text, (10, frame.shape[0] - 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    if not is_aligned:
        cv2.putText(frame, "⚠️ BBOX MISALIGNED!", (10, frame.shape[0] - 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        

class DrawUtils:
    """Classe helper per compatibilità."""
    
    @staticmethod
    def draw_tracked_points(frame, points, color=(0, 255, 0), radius=5, 
                           thickness=-1, labels=True):
        frame_copy = frame.copy()
        left, right = points
        
        cv2.circle(frame_copy, left, radius, color, thickness)
        cv2.circle(frame_copy, right, radius, color, thickness)
        cv2.line(frame_copy, left, right, color, 2)
        
        if labels:
            cv2.putText(frame_copy, 'L', (left[0] - 20, left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_copy, 'R', (right[0] + 10, right[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy