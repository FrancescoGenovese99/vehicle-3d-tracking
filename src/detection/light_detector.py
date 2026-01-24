"""
Light Detector - Rilevamento fari posteriori tramite filtri HSV e blob detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LightCandidate:
    """
    Candidato per un faro rilevato.
    
    Attributes:
        center: Centro (x, y) del blob
        contour: Contorno del blob
        area: Area del blob in pixel
        circularity: Misura di circolarità (0-1)
        bbox: Bounding box (x, y, w, h)
    """
    center: Tuple[int, int]
    contour: np.ndarray
    area: float
    circularity: float
    bbox: Tuple[int, int, int, int]


class LightDetector:
    """
    Detector per fari posteriori e anteriori usando filtri HSV.
    """
    
    def __init__(self, config: Dict):
        """
        Inizializza il detector.
        
        Args:
            config: Dizionario di configurazione (da detection_params.yaml)
        """
        self.config = config
        
        # Range HSV
        hsv_cfg = config.get('hsv_ranges', {})
        red_cfg = hsv_cfg.get('red', {})
        white_cfg = hsv_cfg.get('white', {})
        
        # Range rosso (split su HSV)
        self.red_lower1 = np.array(red_cfg.get('lower1', [0, 100, 100]))
        self.red_upper1 = np.array(red_cfg.get('upper1', [10, 255, 255]))
        self.red_lower2 = np.array(red_cfg.get('lower2', [170, 100, 100]))
        self.red_upper2 = np.array(red_cfg.get('upper2', [180, 255, 255]))
        
        # Range bianco
        self.white_lower = np.array(white_cfg.get('lower', [0, 0, 200]))
        self.white_upper = np.array(white_cfg.get('upper', [180, 30, 255]))
        
        # Parametri blob detection
        blob_cfg = config.get('blob_detection', {})
        self.min_area = blob_cfg.get('min_area', 50)
        self.max_area = blob_cfg.get('max_area', 5000)
        self.min_circularity = blob_cfg.get('min_circularity', 0.4)
        
        # Parametri morfologia
        morph_cfg = config.get('morphology', {})
        kernel_size = morph_cfg.get('kernel_size', [5, 5])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size))
        self.open_iterations = morph_cfg.get('open_iterations', 1)
        self.close_iterations = morph_cfg.get('close_iterations', 1)
    
    def detect_red_lights(self, frame: np.ndarray) -> np.ndarray:
        """
        Crea maschera per luci rosse (fari posteriori).
        
        Args:
            frame: Frame BGR
            
        Returns:
            Maschera binaria con le luci rosse
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Il rosso in HSV è split: 0-10 e 170-180
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def detect_white_lights(self, frame: np.ndarray) -> np.ndarray:
        """
        Crea maschera per luci bianche (targa, freno).
        
        Args:
            frame: Frame BGR
            
        Returns:
            Maschera binaria con le luci bianche
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        return mask
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Applica operazioni morfologiche per pulire la maschera.
        
        Args:
            mask: Maschera binaria
            
        Returns:
            Maschera pulita
        """
        # Opening: rimuove piccoli blob di rumore
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, 
                               iterations=self.open_iterations)
        
        # Closing: chiude piccoli buchi
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel,
                               iterations=self.close_iterations)
        
        return mask
    
    def find_light_candidates(self, mask: np.ndarray) -> List[LightCandidate]:
        """
        Trova candidati per i fari dalla maschera.
        
        Args:
            mask: Maschera binaria
            
        Returns:
            Lista di LightCandidate
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtra per area
            if not (self.min_area < area < self.max_area):
                continue
            
            # Calcola circolarità
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filtra per circolarità (i fari tendono ad essere circolari/ellittici)
            if circularity < self.min_circularity:
                continue
            
            # Calcola centro (momento)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Bounding box
            bbox = cv2.boundingRect(cnt)
            
            candidates.append(LightCandidate(
                center=(cx, cy),
                contour=cnt,
                area=area,
                circularity=circularity,
                bbox=bbox
            ))
        
        return candidates
    
    def detect_tail_lights(self, frame: np.ndarray, 
                          combine_masks: bool = True) -> Tuple[List[LightCandidate], np.ndarray]:
        """
        Rileva i fari posteriori in un frame.
        
        Args:
            frame: Frame BGR
            combine_masks: Se True, combina maschere rosso e bianco
            
        Returns:
            Tuple (candidati, maschera_combinata)
        """
        # Crea maschere
        red_mask = self.detect_red_lights(frame)
        white_mask = self.detect_white_lights(frame)
        
        # Combina maschere
        if combine_masks:
            combined_mask = cv2.bitwise_or(red_mask, white_mask)
        else:
            combined_mask = red_mask
        
        # Applica morfologia
        combined_mask = self.apply_morphology(combined_mask)
        
        # Trova candidati
        candidates = self.find_light_candidates(combined_mask)
        
        return candidates, combined_mask
    
    def visualize_detection(self, frame: np.ndarray, candidates: List[LightCandidate],
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualizza i candidati rilevati sul frame.
        
        Args:
            frame: Frame originale
            candidates: Lista di candidati rilevati
            mask: Maschera opzionale da mostrare
            
        Returns:
            Frame con visualizzazione
        """
        vis_frame = frame.copy()
        
        for candidate in candidates:
            # Disegna contorno
            cv2.drawContours(vis_frame, [candidate.contour], -1, (0, 255, 0), 2)
            
            # Disegna centro
            cv2.circle(vis_frame, candidate.center, 5, (0, 0, 255), -1)
            
            # Disegna bounding box
            x, y, w, h = candidate.bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Aggiungi testo con info
            text = f"A:{candidate.area:.0f} C:{candidate.circularity:.2f}"
            cv2.putText(vis_frame, text, (candidate.center[0] + 10, candidate.center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostra maschera se fornita
        if mask is not None:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis_frame = np.hstack([vis_frame, mask_colored])
        
        return vis_frame