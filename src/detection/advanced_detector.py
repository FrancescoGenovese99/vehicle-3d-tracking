"""
Advanced Detector - ADAPTIVE VERSION with CANNY EDGE DETECTION
Sistema adattivo senza vincoli posizionali, ottimizzato per fari rossi saturi

STRATEGIA:
1. Detection HSV: rosso saturo + alta luminosità (cattura fari anche "quasi bianchi")
2. Nessun vincolo posizionale (funziona ovunque nel frame)
3. Anchor points su EDGE LATERALI a metà altezza del faro
4. Canny edge detection per trovare bordo preciso faro/riflesso
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class VehicleKeypoints:
    """Punti chiave rilevati del veicolo."""
    tail_lights: np.ndarray  # Shape (2, 2): [[left_x, left_y], [right_x, right_y]]
    plate_corners: Optional[Dict[str, Tuple[int, int]]]
    plate_center: Tuple[int, int]
    confidence: float
    templates: Optional[List[np.ndarray]] = None


class AdvancedDetector:
    """
    Detector ADATTIVO con edge detection precisa.
    
    INNOVAZIONI:
    - Nessun vincolo su posizione Y (funziona ovunque nel frame)
    - Detection basata su colore ROSSO + luminosità ALTA
    - Anchor points su EDGE LATERALI (canny-refined)
    - Altezza ADATTIVA (mid-point tra 1/4 e 3/4 del faro)
    """
    
    def __init__(self, config: Dict = None):
        """Inizializza detector."""
        # ===== PARAMETRI HSV ROSSO (OTTIMIZZATI) =====
        # Range per rosso SATURO (anche quando sembra bianco per saturazione)
        self.red_h_lower1 = 0
        self.red_h_upper1 = 10
        self.red_h_lower2 = 170
        self.red_h_upper2 = 180
        
        # Saturazione: PERMISSIVA (anche bassa saturazione = quasi bianco)
        self.red_s_lower = 100  # Molto permissivo
        self.red_s_upper = 255
        
        # Valore: ALTA luminosità (fari molto luminosi)
        self.red_v_lower = 180  # Molto luminoso
        self.red_v_upper = 200
        
        # ===== PARAMETRI BLOB DETECTION =====
        self.min_contour_area = 150  # Pixel² (permissivo per fari lontani)
        self.max_contour_area = 8000  # Pixel² (permissivo per fari vicini)
        self.min_vertical_ratio = 1.2  # h/w > 1.2 (fari verticali)
        
        # ===== PARAMETRI CANNY =====
        self.canny_low = 50
        self.canny_high = 150
        
        # ===== PARAMETRI ANCHOR POINT =====
        self.mid_height_min = 0.25  # 1/4 dell'altezza
        self.mid_height_max = 0.75  # 3/4 dell'altezza
        
        # Template extraction
        self.template_size = 35  # Template grande per catturare edge
        
        # Parametri targa (invariati)
        self.v_plate_low = 150
        self.v_plate_high = 240
        
        # Override con config
        if config:
            self.red_v_lower = config.get('v_lower', self.red_v_lower)
            self.v_plate_low = config.get('v_plate_low', self.v_plate_low)
            self.v_plate_high = config.get('v_plate_high', self.v_plate_high)
        
        print("🔍 AdvancedDetector ADAPTIVE initialized:")
        print(f"  HSV Red: H=[{self.red_h_lower1}-{self.red_h_upper1}, {self.red_h_lower2}-{self.red_h_upper2}], S>={self.red_s_lower}, V>={self.red_v_lower}")
        print(f"  Area: {self.min_contour_area}-{self.max_contour_area}px²")
        print(f"  Anchor: mid-height [{self.mid_height_min}-{self.mid_height_max}]")
    
    def _create_red_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Crea maschera per fari rossi SATURI + alta luminosità.
        
        Cattura:
        - Rosso puro
        - Rosso saturo che appare quasi bianco
        - Fari molto luminosi di notte
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Range rosso (split HSV: 0-10 e 170-180)
        lower1 = np.array([self.red_h_lower1, self.red_s_lower, self.red_v_lower])
        upper1 = np.array([self.red_h_upper1, self.red_s_upper, self.red_v_upper])
        
        lower2 = np.array([self.red_h_lower2, self.red_s_lower, self.red_v_lower])
        upper2 = np.array([self.red_h_upper2, self.red_s_upper, self.red_v_upper])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combina maschere
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def _find_edge_point_with_canny(
        self,
        frame_gray: np.ndarray,
        contour: np.ndarray,
        target_y: int,
        search_direction: str
    ) -> Optional[Tuple[int, int]]:
        """
        Trova edge preciso del faro usando Canny edge detection.
        MIGLIORATO: Filtra solo edge VERTICALI per evitare bordi inferiori.
        """
        # Bounding box del contorno
        x, y, w, h = cv2.boundingRect(contour)
        
        # ROI attorno al contorno (con padding)
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame_gray.shape[1], x + w + padding)
        y2 = min(frame_gray.shape[0], y + h + padding)
        
        roi = frame_gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Canny edge detection
        edges = cv2.Canny(roi, self.canny_low, self.canny_high)
        
        # ===== AGGIUNTO: Calcola gradiente per filtrare edge verticali =====
        # Calcola Sobel per gradiente X e Y
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitudine gradiente
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Angolo gradiente (in radianti)
        grad_angle = np.arctan2(sobely, sobelx)
        
        # Converti a gradi
        grad_angle_deg = np.degrees(grad_angle)
        
        # Maschera per edge VERTICALI (±30° dalla verticale)
        # Edge verticali hanno gradiente orizzontale (angle ≈ 0° o ±180°)
        vertical_mask = (
            (np.abs(grad_angle_deg) < 30) |           # Edge verticale destra
            (np.abs(grad_angle_deg - 180) < 30) |     # Edge verticale sinistra
            (np.abs(grad_angle_deg + 180) < 30)
        )
        
        # Combina edge detection con filtro verticale
        edges_vertical = edges.copy()
        edges_vertical[~vertical_mask] = 0
        # ===== FINE AGGIUNTO =====
        
        # Converti target_y in coordinate ROI
        target_y_roi = target_y - y1
        
        # Cerca edge nella riga target (±3px di tolleranza)
        y_min_roi = max(0, target_y_roi - 3)
        y_max_roi = min(edges_vertical.shape[0], target_y_roi + 3)
        
        # Estrai striscia orizzontale
        edge_strip = edges_vertical[y_min_roi:y_max_roi, :]  # USA edges_vertical
        
        if edge_strip.size == 0:
            return None
        
        # Trova edge points nella striscia
        edge_coords = np.column_stack(np.where(edge_strip > 0))
        
        if len(edge_coords) == 0:
            return None
        
        # Converti coordinate strip → ROI
        edge_points_roi = edge_coords.copy()
        edge_points_roi[:, 0] += y_min_roi
        edge_points_roi = edge_points_roi[:, [1, 0]]  # Swap a (x, y)
        
        # Seleziona edge point in base alla direzione
        if search_direction == 'left':
            idx = edge_points_roi[:, 0].argmin()
        else:  # 'right'
            idx = edge_points_roi[:, 0].argmax()
        
        edge_point_roi = edge_points_roi[idx]
        
        # Converti ROI → coordinate globali
        edge_x = int(edge_point_roi[0] + x1)
        edge_y = int(edge_point_roi[1] + y1)
        
        return (edge_x, edge_y)
    
    def _get_anchor_point_left(
        self,
        contour: np.ndarray,
        frame_gray: np.ndarray
    ) -> Tuple[int, int]:
        """
        Anchor point per faro SINISTRO.
        
        STRATEGIA:
        1. Calcola mid-height del faro (tra 1/4 e 3/4 altezza)
        2. Usa Canny per trovare edge SINISTRO a quella altezza
        3. Fallback: punto più a sinistra del contorno a mid-height
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        # Mid-height adattiva
        mid_ratio = (self.mid_height_min + self.mid_height_max) / 2
        target_y = int(y + h * mid_ratio)
        
        # STRATEGIA 1: Canny edge detection
        edge_point = self._find_edge_point_with_canny(
            frame_gray, contour, target_y, search_direction='left'
        )
        
        if edge_point is not None:
            print(f"  [L] Canny edge: {edge_point}")
            return edge_point
        
        # STRATEGIA 2: Fallback geometrico
        # Trova punti del contorno nella fascia mid-height (±10% altezza)
        y_min = int(y + h * self.mid_height_min)
        y_max = int(y + h * self.mid_height_max)
        
        mid_points = contour[(contour[:, 0, 1] >= y_min) & (contour[:, 0, 1] <= y_max)]
        
        if len(mid_points) > 0:
            # Punto più a SINISTRA nella fascia mid
            leftmost_idx = mid_points[:, 0, 0].argmin()
            anchor = tuple(mid_points[leftmost_idx, 0])
            print(f"  [L] Geometric fallback: {anchor}")
            return anchor
        
        # STRATEGIA 3: Ultimo fallback (centro-sinistra bbox)
        anchor = (x, y + h // 2)
        print(f"  [L] Bbox fallback: {anchor}")
        return anchor
    
    def _get_anchor_point_right(
        self,
        contour: np.ndarray,
        frame_gray: np.ndarray
    ) -> Tuple[int, int]:
        """
        Anchor point per faro DESTRO.
        
        STRATEGIA: Simmetrica a faro sinistro, ma cerca edge DESTRO.
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        # Mid-height adattiva
        mid_ratio = (self.mid_height_min + self.mid_height_max) / 2
        target_y = int(y + h * mid_ratio)
        
        # STRATEGIA 1: Canny edge detection
        edge_point = self._find_edge_point_with_canny(
            frame_gray, contour, target_y, search_direction='right'
        )
        
        if edge_point is not None:
            print(f"  [R] Canny edge: {edge_point}")
            return edge_point
        
        # STRATEGIA 2: Fallback geometrico
        y_min = int(y + h * self.mid_height_min)
        y_max = int(y + h * self.mid_height_max)
        
        mid_points = contour[(contour[:, 0, 1] >= y_min) & (contour[:, 0, 1] <= y_max)]
        
        if len(mid_points) > 0:
            # Punto più a DESTRA nella fascia mid
            rightmost_idx = mid_points[:, 0, 0].argmax()
            anchor = tuple(mid_points[rightmost_idx, 0])
            print(f"  [R] Geometric fallback: {anchor}")
            return anchor
        
        # STRATEGIA 3: Ultimo fallback
        anchor = (x + w, y + h // 2)
        print(f"  [R] Bbox fallback: {anchor}")
        return anchor
    
    def _extract_keypoint_template(self, frame: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Estrae template attorno a un keypoint."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cx, cy = center
        half = self.template_size // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(gray.shape[1], cx + half + 1)
        y2 = min(gray.shape[0], cy + half + 1)
        
        template = gray[y1:y2, x1:x2].copy()
        
        if template.shape[0] < self.template_size or template.shape[1] < self.template_size:
            template = cv2.resize(template, (self.template_size, self.template_size))
        
        return template
    
    def detect_tail_lights(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Rileva fari posteriori ADATTIVAMENTE.
        
        PIPELINE:
        1. Maschera HSV rosso saturo + alta luminosità
        2. Morfologia per cleanup
        3. Trova contorni (filtro solo per area e aspect ratio)
        4. Clustering left/right
        5. Anchor points su edge laterali (Canny-refined)
        
        Returns:
            Numpy array (2, 2): [[left_x, left_y], [right_x, right_y]]
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # === STEP 1: Maschera rosso ===
        mask = self._create_red_mask(frame)
        
        # === STEP 2: Morfologia ===
        # Kernel verticale (enfatizza forme verticali)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # === STEP 3: Trova contorni ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Filtro SOLO per area (nessun vincolo Y!)
            if not (self.min_contour_area < area < self.max_contour_area):
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Filtro aspect ratio (verticale)
            if h == 0 or w / h > 1.0:  # Non troppo largo
                continue
            if h / w < self.min_vertical_ratio:  # Abbastanza verticale
                continue
            
            # Calcola centro per clustering
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            candidates.append((c, cx, cy, area))
        
        if len(candidates) < 2:
            print(f"  ⚠️ Trovati solo {len(candidates)} candidati (servono almeno 2)")
            return None
        
        print(f"  ✓ Trovati {len(candidates)} candidati")
        
        # === STEP 4: SELEZIONE COPPIA OTTIMALE ===
        # Invece di dividere left/right, trova la coppia con:
        # 1. Massima distanza orizzontale
        # 2. Allineamento verticale buono
        # 3. Area simile

        best_pair = None
        best_score = 0
        min_distance = 80  # Distanza minima tra i fari (px)
        max_y_diff = 50     # Max differenza verticale (px)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, x1, y1, a1 = candidates[i]
                c2, x2, y2, a2 = candidates[j]
                
                # Distanza orizzontale
                dx = abs(x2 - x1)
                
                # Differenza verticale
                dy = abs(y2 - y1)
                
                # Ratio area (similarità dimensione)
                area_ratio = min(a1, a2) / max(a1, a2)
                
                # VINCOLI:
                if dx < min_distance:  # Troppo vicini
                    continue
                if dy > max_y_diff:    # Non allineati
                    continue
                if area_ratio < 0.3:   # Troppo diversi
                    continue
                
                # SCORE: privilegia distanza + allineamento + similarità
                score = dx * 2.0 - dy * 1.0 + area_ratio * 50
                
                if score > best_score:
                    best_score = score
                    if x1 < x2:
                        best_pair = (c1, c2)  # (left, right)
                    else:
                        best_pair = (c2, c1)

        if best_pair is None:
            return None

        left_contour, right_contour = best_pair
        
        # === STEP 5: Anchor points con Canny ===
        print("🔍 Detecting anchor points (EDGE MID-HEIGHT):")
        
        left_point = self._get_anchor_point_left(left_contour, gray)
        right_point = self._get_anchor_point_right(right_contour, gray)
        
        tail_lights = np.array([left_point, right_point], dtype=np.float32)
        
        print(f"  ✅ Anchor points: L={left_point}, R={right_point}")
        
        return tail_lights
    
    def detect_tail_lights_with_templates(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Rileva fari + estrae template.
        
        Returns:
            Tuple (tail_lights, templates) o (None, None)
        """
        tail_lights = self.detect_tail_lights(frame)
        
        if tail_lights is None:
            return None, None
        
        templates = []
        for i in range(2):
            center = tuple(map(int, tail_lights[i]))
            template = self._extract_keypoint_template(frame, center)
            templates.append(template)
        
        return tail_lights, templates
    
    def detect_plate_corners(self, frame: np.ndarray, 
                            tail_lights: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Rileva angoli targa (invariato dal codice precedente).
        """
        tail_lights_tuple = (tuple(map(int, tail_lights[0])), tuple(map(int, tail_lights[1])))
        
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        
        # Maschera fari
        mask_lights = cv2.inRange(V, self.red_v_lower, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask_lights = cv2.morphologyEx(mask_lights, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_lights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fari_bottom_y = []
        for c in contours:
            if cv2.contourArea(c) >= 50:
                ys = c[:, 0, 1]
                fari_bottom_y.append(max(ys))
        
        if not fari_bottom_y:
            return None
        
        y_base = max(fari_bottom_y)
        
        x1 = int(min(tail_lights_tuple[0][0], tail_lights_tuple[1][0]))
        x2 = int(max(tail_lights_tuple[0][0], tail_lights_tuple[1][0]))
        
        y1 = min(y_base + 20, height - 1)
        y2 = min(y1 + int(0.18 * height), height)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        V_roi = hsv_roi[:, :, 2]
        
        mask_plate = cv2.inRange(V_roi, self.v_plate_low, self.v_plate_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        mask_cluster = np.zeros_like(mask_plate)
        cv2.drawContours(mask_cluster, [largest], -1, 255, -1)
        
        mask_exp = cv2.dilate(mask_cluster,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)),
                             iterations=2)
        
        edges = cv2.Canny(mask_exp, 60, 180)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                               threshold=60, minLineLength=40, maxLineGap=15)
        
        if lines is None or len(lines) < 4:
            return None
        
        verticals, horizontals = [], []
        
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if 60 < angle < 120:
                verticals.append((x1, y1, x2, y2))
            elif angle < 30 or angle > 150:
                horizontals.append((x1, y1, x2, y2))
        
        if len(verticals) < 2 or len(horizontals) < 2:
            return None
        
        def line_midpoint(l):
            return ((l[0] + l[2]) // 2, (l[1] + l[3]) // 2)
        
        left_line = min(verticals, key=lambda l: line_midpoint(l)[0])
        right_line = max(verticals, key=lambda l: line_midpoint(l)[0])
        top_line = min(horizontals, key=lambda l: line_midpoint(l)[1])
        bottom_line = max(horizontals, key=lambda l: line_midpoint(l)[1])
        
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
        
        TL = intersect(L, T)
        TR = intersect(R, T)
        BL = intersect(L, B)
        BR = intersect(R, B)
        
        if None in [TL, TR, BL, BR]:
            return None
        
        # Converti a coordinate globali (aggiungi offset ROI)
        corners = {
            "TL": (TL[0] + x1, TL[1] + y1),
            "TR": (TR[0] + x1, TR[1] + y1),
            "BL": (BL[0] + x1, BL[1] + y1),
            "BR": (BR[0] + x1, BR[1] + y1)
        }
        
        return corners
    
    def detect_all(self, frame: np.ndarray) -> Optional[VehicleKeypoints]:
        """Rileva tutti i punti chiave."""
        tail_lights, templates = self.detect_tail_lights_with_templates(frame)
        if tail_lights is None:
            return None
        
        plate_corners = self.detect_plate_corners(frame, tail_lights)
        
        if plate_corners:
            xs = [x for x, y in plate_corners.values()]
            ys = [y for x, y in plate_corners.values()]
            plate_center = (int(np.mean(xs)), int(np.mean(ys)))
            confidence = 1.0
        else:
            plate_center = (
                int((tail_lights[0][0] + tail_lights[1][0]) / 2),
                int((tail_lights[0][1] + tail_lights[1][1]) / 2) + 50
            )
            confidence = 0.5
        
        return VehicleKeypoints(
            tail_lights=tail_lights,
            plate_corners=plate_corners,
            plate_center=plate_center,
            confidence=confidence,
            templates=templates
        )