"""
Redetection Manager - Gestione re-detection quando il tracking fallisce.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from src.detection.light_detector import LightDetector, LightCandidate
from src.detection.candidate_selector import CandidateSelector


class RedetectionManager:
    """
    Gestisce la re-detection dei fari quando il tracking viene perso.
    """
    
    def __init__(self, detector: LightDetector, selector: CandidateSelector, config: Dict):
        """
        Inizializza il manager.
        
        Args:
            detector: LightDetector per rilevare i fari
            selector: CandidateSelector per selezionare la coppia migliore
            config: Dizionario di configurazione
        """
        self.detector = detector
        self.selector = selector
        
        redetect_cfg = config.get('redetection', {})
        self.confidence_threshold = redetect_cfg.get('confidence_threshold', 0.6)
        self.enable_kalman = redetect_cfg.get('enable_kalman_prediction', True)
        
        # Kalman filter per predire posizione
        self.kalman_filters = None
        if self.enable_kalman:
            self._init_kalman_filters()
    
    def _init_kalman_filters(self):
        """Inizializza Kalman filters per predire posizione fari."""
        # Due Kalman filters (uno per faro sinistro, uno per destro)
        self.kalman_filters = []
        
        for _ in range(2):
            kf = cv2.KalmanFilter(4, 2)  # 4 stati (x, y, vx, vy), 2 misure (x, y)
            
            # Matrice di transizione
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            # Matrice di misura
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            
            # Noise covariance
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            
            self.kalman_filters.append(kf)
    
    def update_kalman(self, centers: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Aggiorna i Kalman filters con le posizioni misurate.
        
        Args:
            centers: Tuple ((left_x, left_y), (right_x, right_y))
        """
        if not self.enable_kalman or self.kalman_filters is None:
            return
        
        for i, center in enumerate(centers):
            measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
            self.kalman_filters[i].correct(measurement)
    
    def predict_position(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Predice la posizione dei fari usando Kalman filter.
        
        Returns:
            Tuple con posizioni predette o None
        """
        if not self.enable_kalman or self.kalman_filters is None:
            return None
        
        predicted_centers = []
        
        for kf in self.kalman_filters:
            prediction = kf.predict()
            x = int(prediction[0])
            y = int(prediction[1])
            predicted_centers.append((x, y))
        
        return tuple(predicted_centers)
    
    def redetect(self, frame: np.ndarray,
                 last_known_centers: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 search_region_scale: float = 2.0) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Esegue re-detection dei fari.
        
        Args:
            frame: Frame corrente
            last_known_centers: Ultime posizioni note (opzionale)
            search_region_scale: Scala della regione di ricerca rispetto alla bbox standard
            
        Returns:
            Tuple con nuovi centri o None se fallisce
        """
        print("🔍 Re-detection in corso...")
        
        # Se abbiamo Kalman, usa la predizione come hint
        predicted = None
        if self.enable_kalman:
            predicted = self.predict_position()
            if predicted:
                print(f"   Predizione Kalman: {predicted}")
        
        # Rileva candidati
        candidates, mask = self.detector.detect_tail_lights(frame)
        
        if not candidates:
            print("   ✗ Nessun candidato trovato")
            return None
        
        print(f"   Trovati {len(candidates)} candidati")
        
        # Se abbiamo una posizione precedente, filtra candidati vicini
        if last_known_centers or predicted:
            reference = predicted if predicted else last_known_centers
            candidates = self.selector.filter_by_previous_position(
                candidates, reference, max_distance=150
            )
            print(f"   Filtrati a {len(candidates)} candidati vicini")
        
        # Seleziona la coppia migliore
        pair = self.selector.select_tail_light_pair(candidates)
        
        if pair is None:
            print("   ✗ Nessuna coppia valida trovata")
            return None
        
        centers = (pair[0].center, pair[1].center)
        print(f"   ✓ Re-detection riuscita: {centers}")
        
        # Aggiorna Kalman con nuova misura
        if self.enable_kalman:
            self.update_kalman(centers)
        
        return centers
    
    def redetect_with_roi(self, frame: np.ndarray,
                         roi: Tuple[int, int, int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Esegue re-detection in una Region of Interest specifica.
        
        Args:
            frame: Frame completo
            roi: Region of Interest (x, y, width, height)
            
        Returns:
            Tuple con centri (in coordinate globali) o None
        """
        x, y, w, h = roi
        
        # Estrai ROI
        roi_frame = frame[y:y+h, x:x+w]
        
        # Rileva nella ROI
        candidates, _ = self.detector.detect_tail_lights(roi_frame)
        
        if not candidates:
            return None
        
        # Converti coordinate da ROI a globali
        for candidate in candidates:
            cx, cy = candidate.center
            candidate.center = (cx + x, cy + y)
        
        # Seleziona coppia
        pair = self.selector.select_tail_light_pair(candidates)
        
        if pair is None:
            return None
        
        return (pair[0].center, pair[1].center)
    
    def compute_search_roi(self, last_centers: Tuple[Tuple[int, int], Tuple[int, int]],
                          frame_shape: Tuple[int, int],
                          scale: float = 2.0) -> Tuple[int, int, int, int]:
        """
        Calcola una ROI di ricerca attorno alle ultime posizioni note.
        
        Args:
            last_centers: Ultime posizioni note
            frame_shape: Shape del frame (height, width)
            scale: Fattore di scala della ROI
            
        Returns:
            ROI (x, y, width, height)
        """
        left, right = last_centers
        
        # Centro della coppia di fari
        center_x = (left[0] + right[0]) // 2
        center_y = (left[1] + right[1]) // 2
        
        # Distanza tra i fari
        distance = abs(right[0] - left[0])
        
        # ROI size basata sulla distanza
        roi_w = int(distance * scale * 1.5)
        roi_h = int(distance * scale)
        
        # Calcola coordinate ROI
        x = max(0, center_x - roi_w // 2)
        y = max(0, center_y - roi_h // 2)
        w = min(roi_w, frame_shape[1] - x)
        h = min(roi_h, frame_shape[0] - y)
        
        return (x, y, w, h)


import cv2  # Import necessario per KalmanFilter