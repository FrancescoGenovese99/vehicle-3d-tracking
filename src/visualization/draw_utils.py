"""
Draw Utils - Funzioni di utilità per disegnare su frame.
VERSIONE 2 - Con motion type overlay e vanishing point visualization
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


def draw_tracking_info(frame: np.ndarray,
                      frame_idx: int,
                      method: str,
                      distance: float,
                      num_points: int) -> None:
    """
    Disegna info tracking in alto a sinistra.
    
    Args:
        frame: Frame da modificare (in-place)
        frame_idx: Numero frame
        method: Nome metodo ("Vanishing Point", "PnP", ecc.)
        distance: Distanza veicolo in metri
        num_points: Numero punti tracciati
    """
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
    """
    Disegna scritta MOTION TYPE in alto al centro con sfondo semi-trasparente.
    
    Args:
        frame: Frame da modificare (in-place)
        motion_type: "TRANSLATION", "ROTATION", o "MIXED"
        bg_alpha: Trasparenza sfondo (0-1)
    """
    h, w = frame.shape[:2]
    
    # Colore testo (SEMPRE GIALLO per alta visibilità)
    text_color = (0, 255, 255)  # Giallo in BGR
    
    # Colore sfondo basato su tipo
    if motion_type == "TRANSLATION":
        bg_color = (0, 128, 0)  # Verde scuro
    elif motion_type == "ROTATION":
        bg_color = (0, 0, 128)  # Rosso scuro
    else:
        bg_color = (0, 82, 128)  # Arancione scuro
    
    # Testo
    text = f"MOTION: {motion_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Calcola dimensioni testo
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Posizione centrata in alto
    x = (w - text_w) // 2
    y = 50
    
    # Crea overlay per sfondo semi-trasparente
    overlay = frame.copy()
    
    # Rettangolo sfondo
    padding = 15
    cv2.rectangle(overlay,
                 (x - padding, y - text_h - padding),
                 (x + text_w + padding, y + baseline + padding),
                 bg_color, -1)
    
    # Blend sfondo
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    # Testo in primo piano (GIALLO)
    cv2.putText(frame, text, (x, y),
               font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_vanishing_point_lines(frame: np.ndarray,
                               lights_prev: np.ndarray,
                               lights_curr: np.ndarray,
                               vanishing_point: Optional[np.ndarray] = None) -> None:
    """
    Disegna le linee delle traiettorie delle luci e il punto di fuga.
    
    Args:
        frame: Frame da modificare (in-place)
        lights_prev: Luci frame precedente [[x1,y1], [x2,y2]]
        lights_curr: Luci frame corrente [[x1,y1], [x2,y2]]
        vanishing_point: Punto di fuga [u, v] (opzionale)
    """
    if lights_prev is None or lights_curr is None:
        return
    
    h, w = frame.shape[:2]
    
    # Colori
    left_color = (255, 255, 0)   # Ciano per luce sinistra
    right_color = (255, 0, 255)  # Magenta per luce destra
    vp_color = (0, 0, 255)       # Rosso per punto di fuga
    
    # Linea traiettoria luce SINISTRA (L_prev → L_curr)
    L_prev = tuple(lights_prev[0].astype(int))
    L_curr = tuple(lights_curr[0].astype(int))
    
    # Estendi la linea fino ai bordi del frame se possibile
    if vanishing_point is not None:
        vp = tuple(vanishing_point.astype(int))
        # Linea da L_curr verso VP (estesa)
        cv2.line(frame, L_curr, vp, left_color, 2, cv2.LINE_AA)
    else:
        # Linea normale tra frame consecutivi
        cv2.line(frame, L_prev, L_curr, left_color, 2, cv2.LINE_AA)
    
    # Linea traiettoria luce DESTRA (R_prev → R_curr)
    R_prev = tuple(lights_prev[1].astype(int))
    R_curr = tuple(lights_curr[1].astype(int))
    
    if vanishing_point is not None:
        vp = tuple(vanishing_point.astype(int))
        # Linea da R_curr verso VP (estesa)
        cv2.line(frame, R_curr, vp, right_color, 2, cv2.LINE_AA)
    else:
        cv2.line(frame, R_prev, R_curr, right_color, 2, cv2.LINE_AA)
    
    # Cerchi sulle posizioni precedenti (più piccoli)
    cv2.circle(frame, L_prev, 3, left_color, -1)
    cv2.circle(frame, R_prev, 3, right_color, -1)
    
    # Cerchi sulle posizioni correnti (più grandi)
    cv2.circle(frame, L_curr, 6, (0, 255, 0), -1)  # Verde
    cv2.circle(frame, R_curr, 6, (0, 255, 0), -1)
    
    # Etichette
    cv2.putText(frame, "L", (L_curr[0] - 20, L_curr[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "R", (R_curr[0] + 10, R_curr[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Disegna punto di fuga se disponibile
    if vanishing_point is not None:
        vp = tuple(vanishing_point.astype(int))
        
        # Verifica se VP è dentro il frame (altrimenti disegna freccia)
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            # VP dentro il frame: disegna croce + cerchio
            cv2.drawMarker(frame, vp, vp_color, 
                         cv2.MARKER_CROSS, 30, 3)
            cv2.circle(frame, vp, 10, vp_color, 2)
            
            # Etichetta VP
            cv2.putText(frame, "VP", (vp[0] + 15, vp[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, vp_color, 2)
        else:
            # VP fuori dal frame: indica direzione con freccia
            # Calcola punto sul bordo nella direzione del VP
            center = (w // 2, h // 2)
            
            # Direzione verso VP
            dx = vp[0] - center[0]
            dy = vp[1] - center[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx /= length
                dy /= length
                
                # Punto sul bordo
                edge_x = center[0] + int(dx * min(w, h) * 0.4)
                edge_y = center[1] + int(dy * min(w, h) * 0.4)
                
                # Freccia verso VP
                cv2.arrowedLine(frame, center, (edge_x, edge_y),
                              vp_color, 3, tipLength=0.3)
                
                # Etichetta
                cv2.putText(frame, "VP →", (edge_x + 10, edge_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, vp_color, 2)


def draw_bbox_3d(frame: np.ndarray,
                projected_points: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2) -> None:
    """
    Disegna la bounding box 3D sul frame.
    
    Args:
        frame: Frame da modificare (in-place)
        projected_points: Punti 2D proiettati (8, 2)
        color: Colore BGR
        thickness: Spessore linee
    """
    if projected_points is None or len(projected_points) != 8:
        return
    
    # Converti a int
    pts = projected_points.astype(int)
    
    # Indici vertici: [0-3] base, [4-7] top
    # 0: post-dx, 1: post-sx, 2: front-sx, 3: front-dx
    
    # Base
    base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in base_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Top
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    for i, j in top_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Pilastri verticali
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in vertical_edges:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness)
    
    # Evidenzia posteriore (dove sono le luci) in ROSSO
    rear_color = (0, 0, 255)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), rear_color, thickness + 1)
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), rear_color, thickness + 1)


class DrawUtils:
    """
    Classe helper per funzioni di disegno (per compatibilità).
    """
    
    @staticmethod
    def draw_tracked_points(frame: np.ndarray,
                           points: Tuple[Tuple[int, int], Tuple[int, int]],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           radius: int = 5,
                           thickness: int = -1,
                           labels: bool = True) -> np.ndarray:
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