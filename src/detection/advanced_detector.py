"""
Advanced Detector - FIX PLATE JUMP & ADAPTIVE THRESHOLD

CORREZIONI:
1. Threshold adaptive basato su velocità movimento
2. Gestione graceful quando detection fallisce
3. Smoothing posizione targa
4. Validazione geometrica più robusta
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy import stats
from collections import deque


@dataclass
class VehicleKeypoints:
    """Punti chiave rilevati del veicolo - MULTI-FEATURE."""
    tail_lights_features: Dict[str, np.ndarray]
    plate_corners: Optional[Dict[str, Tuple[int, int]]]
    plate_bottom: Optional[np.ndarray]
    plate_center: Tuple[int, int]
    confidence: float
    templates: Optional[Dict[str, List[np.ndarray]]] = None


class AdvancedDetector:
    """
    Detector MULTI-FEATURE con ADAPTIVE THRESHOLD per targa.
    """
    
    def __init__(self, config: Dict = None):
        # ===== PARAMETRI HSV ROSSO (FARI) =====
        self.red_h_lower1 = 0
        self.red_h_upper1 = 10
        self.red_h_lower2 = 170
        self.red_h_upper2 = 180
        
        self.red_s_lower = 100
        self.red_s_upper = 255
        
        self.red_v_lower = 210
        self.red_v_upper = 255
        
        # ===== PARAMETRI BLOB DETECTION =====
        self.min_contour_area_ratio = 0.00015
        self.max_contour_area_ratio = 0.01
        self.min_vertical_ratio = 1.2
        self.max_y_ratio = 0.80
        
        # ===== PARAMETRI FEATURE EXTRACTION =====
        self.mid_height_min = 0.30
        self.mid_height_max = 0.70
        
        # ===== PARAMETRI CANNY =====
        self.canny_low = 50
        self.canny_high = 150
        
        # Template
        self.template_size = 35
        
        # ===== PARAMETRI TARGA =====
        self.v_plate_low = 150
        self.v_plate_high = 240
        
        # ===== STABILITÀ TEMPORALE ADAPTIVE =====
        self.prev_plate_bottom = None
        self.prev_plate_corners = None
        
        # ===== FIX: ADAPTIVE THRESHOLD =====
        self.base_max_jump_pixels = 80
        self.max_jump_pixels = self.base_max_jump_pixels
        self.plate_detection_count = 0
        self.warmup_frames = 10
        
        # ===== NUOVO: VELOCITY ESTIMATION =====
        self.plate_center_history = deque(maxlen=5)  # Ultimi 5 centri
        self.plate_velocity = np.array([0.0, 0.0])   # Velocità stimata (px/frame)
        
        # ===== NUOVO: CONFIDENCE TRACKING =====
        self.detection_confidence = 1.0
        self.min_confidence = 0.3
        
        # Override con config
        if config:
            self.red_v_lower = config.get('v_lower', self.red_v_lower)
            self.v_plate_low = config.get('v_plate_low', self.v_plate_low)
            self.v_plate_high = config.get('v_plate_high', self.v_plate_high)
        
        print("🔍 AdvancedDetector MULTI-FEATURE initialized (ADAPTIVE):")
        print(f"  HSV Red: V>={self.red_v_lower}")
        print(f"  Plate: BLOB + RANSAC (V={self.v_plate_low}-{self.v_plate_high})")
        print(f"  Threshold: ADAPTIVE (base={self.base_max_jump_pixels}px)")
        print(f"  Velocity estimation: ENABLED")
    
    def _create_red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Crea maschera fari rossi."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        mask = cv2.inRange(V, self.red_v_lower, 255)
        return mask
    
    def _extract_outer_point_with_canny(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        is_left: bool
    ) -> Tuple[int, int]:
        """Estrae punto OUTER usando Canny."""
        x, y, w, h = bbox
        
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return (x if is_left else x + w, y + h // 2)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        roi_h = y2 - y1
        mid_y_min = int(roi_h * self.mid_height_min)
        mid_y_max = int(roi_h * self.mid_height_max)
        
        candidates = []
        for py in range(mid_y_min, mid_y_max):
            if py >= edges.shape[0]:
                continue
            
            edge_pixels = np.where(edges[py, :] > 0)[0]
            
            if len(edge_pixels) == 0:
                continue
            
            px = edge_pixels[0] if is_left else edge_pixels[-1]
            candidates.append((px, py))
        
        if len(candidates) == 0:
            mid_points = contour[(contour[:, 0, 1] >= y + int(h * self.mid_height_min)) & 
                                 (contour[:, 0, 1] <= y + int(h * self.mid_height_max))]
            
            if len(mid_points) > 0:
                idx = mid_points[:, 0, 0].argmin() if is_left else mid_points[:, 0, 0].argmax()
                return tuple(mid_points[idx, 0])
            else:
                return (x if is_left else x + w, y + h // 2)
        
        candidates = np.array(candidates)
        median_x = int(np.median(candidates[:, 0]))
        median_y = int(np.median(candidates[:, 1]))
        
        global_x = median_x + x1
        global_y = median_y + y1
        
        return (global_x, global_y)
    
    def _extract_feature_points(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        is_left: bool
    ) -> Dict[str, Tuple[int, int]]:
        """Estrae 3 feature points."""
        x, y, w, h = cv2.boundingRect(contour)
        
        features = {}
        
        top_idx = contour[:, 0, 1].argmin()
        features['top'] = tuple(contour[top_idx, 0])
        
        bottom_idx = contour[:, 0, 1].argmax()
        features['bottom'] = tuple(contour[bottom_idx, 0])
        
        features['outer'] = self._extract_outer_point_with_canny(
            frame, contour, (x, y, w, h), is_left
        )
        
        return features
    
    def detect_tail_lights_multifeature(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Rileva fari multi-feature."""
        height, width = frame.shape[:2]
        
        frame_area = width * height
        min_area = self.min_contour_area_ratio * frame_area
        max_area = self.max_contour_area_ratio * frame_area
        max_y = int(height * self.max_y_ratio)
        
        mask = self._create_red_mask(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            
            if not (min_area < area < max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            if y > max_y:
                continue
            
            if w == 0 or h == 0:
                continue
            if w / h > 1.0:
                continue
            if h / w < self.min_vertical_ratio:
                continue
            
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            candidates.append((c, cx, cy, area))
        
        if len(candidates) < 2:
            return None
        
        best_pair = None
        best_score = 0
        min_distance = 80
        max_y_diff = 50
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, x1, y1, a1 = candidates[i]
                c2, x2, y2, a2 = candidates[j]
                
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                area_ratio = min(a1, a2) / max(a1, a2)
                
                if dx < min_distance or dy > max_y_diff or area_ratio < 0.3:
                    continue
                
                score = dx * 2.0 - dy * 1.0 + area_ratio * 50
                
                if score > best_score:
                    best_score = score
                    if x1 < x2:
                        best_pair = (c1, c2)
                    else:
                        best_pair = (c2, c1)
        
        if best_pair is None:
            return None
        
        left_contour, right_contour = best_pair
        
        left_features = self._extract_feature_points(frame, left_contour, is_left=True)
        right_features = self._extract_feature_points(frame, right_contour, is_left=False)
        
        features_dict = {}
        for feature_name in ['top', 'outer', 'bottom']:
            left_pt = left_features[feature_name]
            right_pt = right_features[feature_name]
            features_dict[feature_name] = np.array([left_pt, right_pt], dtype=np.float32)
        
        return features_dict
    
    def _extract_keypoint_template(self, frame: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Estrae template."""
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
    
    def detect_tail_lights_with_templates(self, frame: np.ndarray) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]]:
        """Rileva fari + estrae template."""
        features_dict = self.detect_tail_lights_multifeature(frame)
        
        if features_dict is None:
            return None, None
        
        templates_dict = {}
        for feature_name, points in features_dict.items():
            templates = []
            for i in range(2):
                center = tuple(map(int, points[i]))
                template = self._extract_keypoint_template(frame, center)
                templates.append(template)
            templates_dict[feature_name] = templates
        
        return features_dict, templates_dict
    
    def _fit_line_ransac(self, points: np.ndarray, axis: str = 'horizontal') -> Optional[Tuple[float, float]]:
        """Fitta linea con RANSAC."""
        if len(points) < 2:
            return None
        
        points = np.array(points)
        
        if axis == 'horizontal':
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            
            if np.std(X) < 1e-6:
                return None
        else:
            X = points[:, 1].reshape(-1, 1)
            y = points[:, 0]
            
            if np.std(X) < 1e-6:
                return None
        
        try:
            from sklearn.linear_model import RANSACRegressor
            
            ransac = RANSACRegressor(
                residual_threshold=3.0,
                max_trials=100,
                min_samples=2,
                random_state=42
            )
            
            ransac.fit(X, y)
            
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            return (slope, intercept)
        
        except Exception:
            if axis == 'horizontal':
                slope, intercept, _, _, _ = stats.linregress(points[:, 0], points[:, 1])
            else:
                slope, intercept, _, _, _ = stats.linregress(points[:, 1], points[:, 0])
            
            return (slope, intercept)
    
    def _process_plate_in_roi(
        self,
        frame: np.ndarray,
        roi: np.ndarray,
        roi_offset: Tuple[int, int]
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Processing targa con BLOB + RANSAC."""
        if roi.size == 0 or roi.shape[1] < 50:
            return None
        
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            V_roi = hsv_roi[:, :, 2]
            
            mask_plate = cv2.inRange(V_roi, self.v_plate_low, self.v_plate_high)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)
            
            contours_plate, _ = cv2.findContours(mask_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_plate:
                return None
            
            largest = max(contours_plate, key=cv2.contourArea)
            points = largest.reshape(-1, 2)
            
            if len(points) < 10:
                return None
            
            y_min = points[:, 1].min()
            y_max = points[:, 1].max()
            x_min = points[:, 0].min()
            x_max = points[:, 0].max()
            
            height_plate = y_max - y_min
            width_plate = x_max - x_min
            
            if width_plate < 15 or height_plate < 5:
                return None
            
            top_band = y_min + 0.25 * height_plate
            bottom_band = y_max - 0.25 * height_plate
            left_band = x_min + 0.15 * width_plate
            right_band = x_max - 0.15 * width_plate
            
            top_pts = []
            bottom_pts = []
            left_pts = []
            right_pts = []
            
            for x, y in points:
                if y <= top_band and left_band <= x <= right_band:
                    top_pts.append([x, y])
                
                if y >= bottom_band and left_band <= x <= right_band:
                    bottom_pts.append([x, y])
                
                if x <= left_band and top_band <= y <= bottom_band:
                    left_pts.append([x, y])
                
                if x >= right_band and top_band <= y <= bottom_band:
                    right_pts.append([x, y])
            
            top_line = self._fit_line_ransac(np.array(top_pts), 'horizontal') if len(top_pts) >= 2 else None
            bottom_line = self._fit_line_ransac(np.array(bottom_pts), 'horizontal') if len(bottom_pts) >= 2 else None
            left_line = self._fit_line_ransac(np.array(left_pts), 'vertical') if len(left_pts) >= 2 else None
            right_line = self._fit_line_ransac(np.array(right_pts), 'vertical') if len(right_pts) >= 2 else None
            
            def intersect_h_v(h_line, v_line):
                if h_line is None or v_line is None:
                    return None
                m_h, q_h = h_line
                m_v, q_v = v_line
                denom = 1 - m_h * m_v
                if abs(denom) < 1e-6:
                    return None
                y = (m_h * q_v + q_h) / denom
                x = m_v * y + q_v
                return (int(x), int(y))
            
            TL = intersect_h_v(top_line, left_line)
            TR = intersect_h_v(top_line, right_line)
            BL = intersect_h_v(bottom_line, left_line)
            BR = intersect_h_v(bottom_line, right_line)
            
            if not all([TL, TR, BL, BR]):
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                pts = box.reshape(4, 2)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1)
                
                TL = tuple(pts[np.argmin(s)])
                BR = tuple(pts[np.argmax(s)])
                TR = tuple(pts[np.argmin(d)])
                BL = tuple(pts[np.argmax(d)])
            
            x1, y1 = roi_offset
            corners = {
                "TL": (TL[0] + x1, TL[1] + y1),
                "TR": (TR[0] + x1, TR[1] + y1),
                "BL": (BL[0] + x1, BL[1] + y1),
                "BR": (BR[0] + x1, BR[1] + y1)
            }
            
            return corners
        
        except Exception as e:
            return None
    
    def _estimate_velocity(self) -> np.ndarray:
        """
        Stima velocità targa da history.
        
        Returns:
            Velocità [vx, vy] in px/frame
        """
        if len(self.plate_center_history) < 2:
            return np.array([0.0, 0.0])
        
        # Usa ultimi 3 punti per calcolo velocità
        recent = list(self.plate_center_history)[-3:]
        
        velocities = []
        for i in range(len(recent) - 1):
            delta = np.array(recent[i+1]) - np.array(recent[i])
            velocities.append(delta)
        
        # Media velocità
        avg_velocity = np.mean(velocities, axis=0)
        
        return avg_velocity
    
    def _check_temporal_stability(self, corners: Dict[str, Tuple[int, int]]) -> bool:
        """
        Verifica stabilità ADAPTIVE.
        
        MIGLIORAMENTI:
        1. Threshold basato su velocità stimata
        2. Warmup progressivo
        3. Confidence decay
        """
        if self.prev_plate_corners is None:
            return True
        
        # Calcola centro
        xs = [x for x, y in corners.values()]
        ys = [y for x, y in corners.values()]
        current_center = np.array([np.mean(xs), np.mean(ys)])
        
        prev_xs = [x for x, y in self.prev_plate_corners.values()]
        prev_ys = [y for x, y in self.prev_plate_corners.values()]
        prev_center = np.array([np.mean(prev_xs), np.mean(prev_ys)])
        
        delta = np.linalg.norm(current_center - prev_center)
        
        # ===== FIX: ADAPTIVE THRESHOLD =====
        # Base threshold
        if self.plate_detection_count < self.warmup_frames:
            base_threshold = self.base_max_jump_pixels * 2.0  # Warmup
        else:
            base_threshold = self.base_max_jump_pixels
        
        # Stima velocità
        velocity = self._estimate_velocity()
        expected_movement = np.linalg.norm(velocity)
        
        # Threshold adattivo: base + expected_movement * safety_factor
        threshold = base_threshold + expected_movement * 2.0
        
        # Clamp threshold
        threshold = min(threshold, 200.0)  # Max 200px
        
        if delta > threshold:
            print(f"⚠️ Plate JUMP: {delta:.1f}px > {threshold:.1f}px (vel={expected_movement:.1f}px/frame)")
            
            # Decay confidence
            self.detection_confidence *= 0.7
            
            return False
        
        # Aggiorna confidence
        self.detection_confidence = min(1.0, self.detection_confidence * 1.1)
        
        # Aggiorna history
        self.plate_center_history.append(current_center)
        
        self.plate_detection_count += 1
        return True
    
    def detect_plate_corners(
        self,
        frame: np.ndarray,
        tail_lights_features: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Rileva angoli targa con ADAPTIVE THRESHOLD."""
        height, width = frame.shape[:2]
        
        bottom_points = tail_lights_features.get('bottom')
        outer_points = tail_lights_features.get('outer')
        
        if bottom_points is None or outer_points is None:
            return None
        
        fari_bottom_y = max(bottom_points[0][1], bottom_points[1][1])
        
        lights_x_min = min(outer_points[0][0], outer_points[1][0])
        lights_x_max = max(outer_points[0][0], outer_points[1][0])
        lights_width = lights_x_max - lights_x_min
        
        margin_x = int(lights_width * 0.10)
        x1 = max(0, int(lights_x_min - margin_x))
        x2 = min(width, int(lights_x_max + margin_x))
        
        scale = lights_width
        y1 = int(fari_bottom_y + 0.15 * scale)
        y2 = int(fari_bottom_y + 0.45 * scale)
        
        y1 = max(0, y1)
        y2 = min(height, y2)
        
        roi = frame[y1:y2, x1:x2]
        
        corners = self._process_plate_in_roi(frame, roi, (x1, y1))
        
        if corners is None:
            # ===== FIX: PREDICTION SE DETECTION FALLISCE =====
            if self.prev_plate_corners is not None and len(self.plate_center_history) >= 2:
                # Predici usando velocità
                velocity = self._estimate_velocity()
                
                predicted_corners = {}
                for key, (px, py) in self.prev_plate_corners.items():
                    predicted_corners[key] = (
                        int(px + velocity[0]),
                        int(py + velocity[1])
                    )
                
                print(f"⚡ Using predicted plate position (vel={velocity})")
                return predicted_corners
            
            if self.prev_plate_corners is not None:
                print("⚡ Using previous plate position (detection failed)")
                return self.prev_plate_corners
            
            return None
        
        # Verifica stabilità
        if not self._check_temporal_stability(corners):
            # Salto troppo grande: usa predizione o precedente
            if self.prev_plate_corners is not None:
                if self.detection_confidence < self.min_confidence:
                    # Confidence troppo bassa: accetta nuova detection
                    print(f"✓ Accepting new detection (low confidence: {self.detection_confidence:.2f})")
                    self.prev_plate_corners = corners
                    self.detection_confidence = 0.5
                    return corners
                else:
                    # Usa precedente
                    return self.prev_plate_corners
        
        # Aggiorna stato
        self.prev_plate_corners = corners
        
        return corners
    
    def update_plate_bottom_only(
        self,
        frame: np.ndarray,
        tail_lights_features: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Aggiorna plate bottom con ROBUSTEZZA."""
        plate_corners = self.detect_plate_corners(frame, tail_lights_features)
        
        if plate_corners is None:
            return self.prev_plate_bottom
        
        BL = np.array(plate_corners['BL'], dtype=np.float32)
        BR = np.array(plate_corners['BR'], dtype=np.float32)
        
        plate_bottom = np.array([BL, BR], dtype=np.float32)
        
        self.prev_plate_bottom = plate_bottom
        
        return plate_bottom
    
    def detect_all_multifeature(self, frame: np.ndarray) -> Optional[VehicleKeypoints]:
        """Rileva tutti i punti chiave."""
        features_dict, templates_dict = self.detect_tail_lights_with_templates(frame)
        if features_dict is None:
            return None
        
        plate_corners = self.detect_plate_corners(frame, features_dict)
        
        plate_bottom = None
        if plate_corners:
            BL = np.array(plate_corners['BL'], dtype=np.float32)
            BR = np.array(plate_corners['BR'], dtype=np.float32)
            plate_bottom = np.array([BL, BR], dtype=np.float32)
            
            xs = [x for x, y in plate_corners.values()]
            ys = [y for x, y in plate_corners.values()]
            plate_center = (int(np.mean(xs)), int(np.mean(ys)))
            
            # Aggiorna history
            self.plate_center_history.append(np.array(plate_center))
            
            confidence = self.detection_confidence
        else:
            outer = features_dict['outer']
            plate_center = (
                int((outer[0][0] + outer[1][0]) / 2),
                int((outer[0][1] + outer[1][1]) / 2) + 50
            )
            confidence = 0.5
        
        return VehicleKeypoints(
            tail_lights_features=features_dict,
            plate_corners=plate_corners,
            plate_bottom=plate_bottom,
            plate_center=plate_center,
            confidence=confidence,
            templates=templates_dict
        )
    
    def detect_tail_lights(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Backward compatibility."""
        features_dict = self.detect_tail_lights_multifeature(frame)
        if features_dict is None:
            return None
        return features_dict.get('outer')