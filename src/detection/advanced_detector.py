"""
Advanced Detector - Detection robusta di fari e targa usando luminosità.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class VehicleKeypoints:
    """
    Punti chiave rilevati del veicolo.
    
    Attributes:
        tail_lights: Tuple con centri dei 2 fari ((left_x, left_y), (right_x, right_y))
        plate_corners: Dict con 4 angoli targa {"TL": (x,y), "TR": (x,y), "BL": (x,y), "BR": (x,y)}
        plate_center: Centro della targa (x, y)
        confidence: Score di confidenza (0-1)
    """
    tail_lights: Tuple[Tuple[int, int], Tuple[int, int]]
    plate_corners: Dict[str, Tuple[int, int]]
    plate_center: Tuple[int, int]
    confidence: float


class AdvancedDetector:
    """
    Detector avanzato che usa luminosità invece di colore HSV.
    Più robusto per scene notturne.
    """
    
    def __init__(self, config: Dict = None):
        """
        Inizializza detector.
        
        Args:
            config: Configurazione opzionale
        """
        # Parametri fari
        self.v_lower = 210
        self.max_y_ratio = 0.80
        self.min_vertical_ratio = 1.2
        self.min_contour_area_ratio = 0.00015
        self.y_tolerance = 5
        
        # Parametri targa
        self.v_plate_low = 150
        self.v_plate_high = 240
        
        # Override con config se fornita
        if config:
            self.v_lower = config.get('v_lower', self.v_lower)
            self.max_y_ratio = config.get('max_y_ratio', self.max_y_ratio)
            self.v_plate_low = config.get('v_plate_low', self.v_plate_low)
            self.v_plate_high = config.get('v_plate_high', self.v_plate_high)
    
    def detect_tail_lights(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Rileva i centri dei fari posteriori.
        
        Args:
            frame: Frame BGR
            
        Returns:
            Tuple ((left_x, left_y), (right_x, right_y)) o None
        """
        height, width = frame.shape[:2]
        
        # Estrai canale V (luminosità)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        
        # Maschera luminosità alta
        mask = cv2.inRange(V, self.v_lower, 255)
        
        # Morfologia (enfatizza forme verticali)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Trova contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = self.min_contour_area_ratio * width * height
        max_y = int(height * self.max_y_ratio)
        
        candidates = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Filtra per posizione e forma
            if y > max_y:
                continue
            if w / h > 1.0:  # Non troppo largo
                continue
            if h / w < self.min_vertical_ratio:  # Abbastanza verticale
                continue
            
            # Calcola centro
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            candidates.append((c, cx, cy))
        
        if len(candidates) < 2:
            return None
        
        # Ordina per x e dividi in 2 cluster (sinistro/destro)
        candidates = sorted(candidates, key=lambda p: p[1])
        mid = len(candidates) // 2
        clusters = [candidates[:mid], candidates[mid:]]
        
        # Calcola centro pesato per ogni cluster
        fari = []
        for cluster in clusters:
            xs, ys, areas = [], [], []
            for c, cx, cy in cluster:
                area = cv2.contourArea(c)
                xs.append(cx * area)
                ys.append(cy * area)
                areas.append(area)
            
            if sum(areas) > 0:
                faro_x = int(sum(xs) / sum(areas))
                faro_y = int(sum(ys) / sum(areas))
                fari.append((faro_x, faro_y))
        
        if len(fari) == 2:
            return tuple(fari)
        
        return None
    
    def detect_plate_corners(self, frame: np.ndarray, 
                            tail_lights: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Rileva i 4 angoli della targa usando i fari come guida.
        
        Args:
            frame: Frame BGR
            tail_lights: Centri dei fari
            
        Returns:
            Dict con 4 angoli {"TL", "TR", "BL", "BR"} o None
        """
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        
        # === STEP 1: Trova ROI targa ===
        # Usa i fari per guidare la ricerca
        mask_lights = cv2.inRange(V, self.v_lower, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_lights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Trova base verticale fari (punto più basso)
        min_area = self.min_contour_area_ratio * width * height
        fari_bottom_y = []
        
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                ys = c[:, 0, 1]
                fari_bottom_y.append(max(ys))
        
        if not fari_bottom_y:
            return None
        
        y_base = max(fari_bottom_y)
        
        # Limiti orizzontali dalla posizione fari
        x1 = min(tail_lights[0][0], tail_lights[1][0])
        x2 = max(tail_lights[0][0], tail_lights[1][0])
        
        # ROI verticale sotto i fari
        y1 = min(y_base + 20, height - 1)
        y2 = min(y1 + int(0.18 * height), height)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # === STEP 2: Maschera luce targa ===
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        V_roi = hsv_roi[:, :, 2]
        
        mask_plate = cv2.inRange(V_roi, self.v_plate_low, self.v_plate_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Cluster luce targa (più grande)
        largest = max(contours, key=cv2.contourArea)
        mask_cluster = np.zeros_like(mask_plate)
        cv2.drawContours(mask_cluster, [largest], -1, 255, -1)
        
        # Espandi per catturare bordi
        mask_exp = cv2.dilate(mask_cluster,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)),
                             iterations=2)
        
        # === STEP 3: Edge detection ===
        edges = cv2.Canny(mask_exp, 60, 180)
        
        # === STEP 4: Hough lines ===
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                               threshold=60, minLineLength=40, maxLineGap=15)
        
        if lines is None or len(lines) < 4:
            return None
        
        # Classifica linee in verticali/orizzontali
        verticals, horizontals = [], []
        
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if 60 < angle < 120:  # Verticale
                verticals.append((x1, y1, x2, y2))
            elif angle < 30 or angle > 150:  # Orizzontale
                horizontals.append((x1, y1, x2, y2))
        
        if len(verticals) < 2 or len(horizontals) < 2:
            return None
        
        # === STEP 5: Trova 4 lati ===
        def line_midpoint(l):
            return ((l[0] + l[2]) // 2, (l[1] + l[3]) // 2)
        
        left_line = min(verticals, key=lambda l: line_midpoint(l)[0])
        right_line = max(verticals, key=lambda l: line_midpoint(l)[0])
        top_line = min(horizontals, key=lambda l: line_midpoint(l)[1])
        bottom_line = max(horizontals, key=lambda l: line_midpoint(l)[1])
        
        # === STEP 6: Fit linee e trova intersezioni ===
        def fit_line(line):
            x1, y1, x2, y2 = line
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            return vx[0], vy[0], x0[0], y0[0]
        
        def intersect(l1, l2):
            vx1, vy1, x1, y1 = l1
            vx2, vy2, x2, y2 = l2
            
            A = np.array([[vx1, -vx2], [vy1, -vy2]])
            B = np.array([[x2 - x1], [y2 - y1]])
            
            try:
                t = np.linalg.lstsq(A, B, rcond=None)[0]
                xi = x1 + t[0] * vx1
                yi = y1 + t[0] * vy1
                return int(xi), int(yi)
            except:
                return None
        
        L = fit_line(left_line)
        R = fit_line(right_line)
        T = fit_line(top_line)
        B = fit_line(bottom_line)
        
        # Calcola 4 angoli (coordinate ROI)
        TL = intersect(L, T)
        TR = intersect(R, T)
        BL = intersect(L, B)
        BR = intersect(R, B)
        
        if None in [TL, TR, BL, BR]:
            return None
        
        # Converti da coordinate ROI a coordinate globali
        corners = {
            "TL": (TL[0] + x1, TL[1] + y1),
            "TR": (TR[0] + x1, TR[1] + y1),
            "BL": (BL[0] + x1, BL[1] + y1),
            "BR": (BR[0] + x1, BR[1] + y1)
        }
        
        return corners
    
    def detect_all(self, frame: np.ndarray) -> Optional[VehicleKeypoints]:
        """
        Rileva tutti i punti chiave del veicolo.
        
        Args:
            frame: Frame BGR
            
        Returns:
            VehicleKeypoints o None
        """
        # Rileva fari
        tail_lights = self.detect_tail_lights(frame)
        if tail_lights is None:
            return None
        
        # Rileva targa
        plate_corners = self.detect_plate_corners(frame, tail_lights)
        
        # Calcola centro targa
        if plate_corners:
            xs = [x for x, y in plate_corners.values()]
            ys = [y for x, y in plate_corners.values()]
            plate_center = (int(np.mean(xs)), int(np.mean(ys)))
            confidence = 1.0
        else:
            # Se non trova targa, stima dal centro fari
            plate_center = (
                int((tail_lights[0][0] + tail_lights[1][0]) / 2),
                int((tail_lights[0][1] + tail_lights[1][1]) / 2) + 50
            )
            confidence = 0.5
        
        return VehicleKeypoints(
            tail_lights=tail_lights,
            plate_corners=plate_corners,
            plate_center=plate_center,
            confidence=confidence
        )