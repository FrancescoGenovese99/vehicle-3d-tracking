"""
Tracker - Tracking temporale multi-frame dei fari con OpenCV trackers.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TrackerType(Enum):
    """Tipi di tracker disponibili."""
    CSRT = "CSRT"
    KCF = "KCF"
    MOSSE = "MOSSE"


class LightTracker:
    """
    Tracker per seguire i fari posteriori frame-by-frame.
    """
    
    def __init__(self, config: Dict):
        """
        Inizializza il tracker.
        
        Args:
            config: Dizionario di configurazione (da detection_params.yaml)
        """
        tracking_cfg = config.get('tracking', {})
        
        # Tipo di tracker
        tracker_name = tracking_cfg.get('tracker_type', 'CSRT')
        self.tracker_type = TrackerType[tracker_name]
        
        # Parametri
        self.bbox_padding = tracking_cfg.get('bbox_padding', 20)
        self.max_frames_lost = tracking_cfg.get('max_frames_lost', 10)
        
        # Stato tracking
        self.trackers = []  # Lista di tracker OpenCV
        self.current_centers = None  # Centri correnti
        self.frames_since_detection = 0
        self.is_initialized = False
    
    def _create_tracker(self) -> cv2.Tracker:
        """
        Crea un nuovo tracker OpenCV.
        
        Returns:
            Tracker OpenCV
        """
        if self.tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == TrackerType.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        else:
            raise ValueError(f"Tracker type non supportato: {self.tracker_type}")
    
    def _point_to_bbox(self, point: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Converte un punto centrale in una bounding box.
        
        Args:
            point: Centro (x, y)
            
        Returns:
            Bounding box (x, y, w, h)
        """
        x, y = point
        half_size = self.bbox_padding
        
        return (
            max(0, x - half_size),
            max(0, y - half_size),
            2 * half_size,
            2 * half_size
        )
    
    def _bbox_to_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Converte una bounding box al suo centro.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Centro (x, y)
        """
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def initialize(self, frame: np.ndarray, 
                  initial_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Inizializza il tracking con i punti iniziali.
        
        Args:
            frame: Primo frame
            initial_points: Tuple ((left_x, left_y), (right_x, right_y))
        """
        self.trackers = []
        
        for point in initial_points:
            tracker = self._create_tracker()
            bbox = self._point_to_bbox(point)
            tracker.init(frame, bbox)
            self.trackers.append(tracker)
        
        self.current_centers = initial_points
        self.frames_since_detection = 0
        self.is_initialized = True
        
        print(f"✓ Tracking inizializzato con {len(self.trackers)} fari")
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Aggiorna il tracking sul nuovo frame.
        
        Args:
            frame: Frame corrente
            
        Returns:
            Tuple (success, centers) dove:
            - success: True se il tracking ha successo
            - centers: Tuple con i centri aggiornati o None
        """
        if not self.is_initialized:
            return False, None
        
        updated_centers = []
        success_count = 0
        
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            
            if success:
                center = self._bbox_to_point(bbox)
                updated_centers.append(center)
                success_count += 1
            else:
                # Tracker fallito, usa ultima posizione nota
                if self.current_centers and len(updated_centers) < len(self.current_centers):
                    updated_centers.append(self.current_centers[len(updated_centers)])
        
        # Considera il tracking riuscito se almeno 1 tracker funziona
        if success_count > 0:
            self.current_centers = tuple(updated_centers)
            self.frames_since_detection = 0
            return True, self.current_centers
        else:
            self.frames_since_detection += 1
            return False, self.current_centers
    
    def reinitialize(self, frame: np.ndarray, 
                    new_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Reinizializza il tracking con nuovi punti (dopo re-detection).
        
        Args:
            frame: Frame corrente
            new_points: Nuovi centri da trackare
        """
        print(f"↻ Reinizializzazione tracking al frame")
        self.initialize(frame, new_points)
    
    def needs_redetection(self) -> bool:
        """
        Controlla se è necessaria una re-detection.
        
        Returns:
            True se il tracking è stato perso per troppi frame
        """
        return self.frames_since_detection > self.max_frames_lost
    
    def get_current_centers(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Ottiene i centri correnti.
        
        Returns:
            Tuple con i centri correnti o None
        """
        return self.current_centers
    
    def reset(self):
        """Reset completo del tracker."""
        self.trackers = []
        self.current_centers = None
        self.frames_since_detection = 0
        self.is_initialized = False


class MultiObjectTracker:
    """
    Tracker avanzato che gestisce più oggetti con ID persistenti.
    Opzionale per funzionalità future.
    """
    
    def __init__(self, max_disappeared: int = 10):
        """
        Args:
            max_disappeared: Numero massimo di frame senza detection prima di rimuovere un oggetto
        """
        self.next_object_id = 0
        self.objects = {}  # {id: center}
        self.disappeared = {}  # {id: count}
        self.max_disappeared = max_disappeared
    
    def register(self, center: Tuple[int, int]) -> int:
        """
        Registra un nuovo oggetto.
        
        Args:
            center: Centro dell'oggetto
            
        Returns:
            ID assegnato
        """
        obj_id = self.next_object_id
        self.objects[obj_id] = center
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id
    
    def deregister(self, obj_id: int):
        """Rimuove un oggetto dal tracking."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
    
    def update(self, detections: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Aggiorna il tracking con nuove detection.
        
        Args:
            detections: Lista di centri rilevati
            
        Returns:
            Dizionario {object_id: center}
        """
        # Se non ci sono detection, incrementa disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        # Se non ci sono oggetti tracciati, registra tutti
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
            return self.objects
        
        # Associa detection a oggetti esistenti
        object_ids = list(self.objects.keys())
        object_centers = list(self.objects.values())
        
        # Calcola distanze tra tutti gli oggetti e detection
        distances = np.zeros((len(object_centers), len(detections)))
        
        for i, obj_center in enumerate(object_centers):
            for j, det_center in enumerate(detections):
                distances[i, j] = np.linalg.norm(np.array(obj_center) - np.array(det_center))
        
        # Associa usando l'algoritmo greedy (nearest neighbor)
        used_detections = set()
        
        for i in range(len(object_centers)):
            if len(used_detections) >= len(detections):
                break
            
            # Trova detection più vicina
            min_dist = distances[i].min()
            min_idx = distances[i].argmin()
            
            if min_idx not in used_detections:
                obj_id = object_ids[i]
                self.objects[obj_id] = detections[min_idx]
                self.disappeared[obj_id] = 0
                used_detections.add(min_idx)
            else:
                obj_id = object_ids[i]
                self.disappeared[obj_id] += 1
        
        # Registra nuove detection non associate
        for j in range(len(detections)):
            if j not in used_detections:
                self.register(detections[j])
        
        # Rimuovi oggetti scomparsi
        for obj_id in list(self.disappeared.keys()):
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)
        
        return self.objects
# Alias for compatibility with vehicle_localization_system.py
VehicleTracker = LightTracker
