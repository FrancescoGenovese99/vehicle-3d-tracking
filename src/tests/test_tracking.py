"""
Test per i moduli di tracking.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.tracker import LightTracker, TrackerType
from src.utils.config_loader import load_config


@pytest.fixture
def tracking_config():
    """Carica configurazione di tracking."""
    config_path = Path(__file__).parent.parent / "config" / "detection_params.yaml"
    if config_path.exists():
        return load_config(str(config_path))
    else:
        # Config di default per test
        return {
            'tracking': {
                'tracker_type': 'CSRT',
                'bbox_padding': 20,
                'max_frames_lost': 10
            }
        }


@pytest.fixture
def test_frames():
    """Crea una sequenza di frame di test con oggetti in movimento."""
    frames = []
    
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Faro sinistro che si muove
        x_left = 200 + i * 5
        cv2.circle(frame, (x_left, 300), 20, (0, 0, 255), -1)
        
        # Faro destro che si muove
        x_right = 440 + i * 5
        cv2.circle(frame, (x_right, 300), 20, (0, 0, 255), -1)
        
        frames.append(frame)
    
    return frames


class TestLightTracker:
    """Test per LightTracker."""
    
    def test_initialization(self, tracking_config):
        """Test inizializzazione tracker."""
        tracker = LightTracker(tracking_config)
        
        assert tracker is not None
        assert tracker.tracker_type == TrackerType.CSRT
        assert tracker.is_initialized == False
    
    def test_tracker_type_enum(self):
        """Test enum TrackerType."""
        assert TrackerType.CSRT.value == "CSRT"
        assert TrackerType.KCF.value == "KCF"
        assert TrackerType.MOSSE.value == "MOSSE"
    
    def test_initialize_tracker(self, tracking_config, test_frames):
        """Test inizializzazione tracking."""
        tracker = LightTracker(tracking_config)
        
        initial_points = ((200, 300), (440, 300))
        tracker.initialize(test_frames[0], initial_points)
        
        assert tracker.is_initialized == True
        assert len(tracker.trackers) == 2
        assert tracker.current_centers == initial_points
    
    def test_update_tracker(self, tracking_config, test_frames):
        """Test update tracking su frame successivi."""
        tracker = LightTracker(tracking_config)
        
        initial_points = ((200, 300), (440, 300))
        tracker.initialize(test_frames[0], initial_points)
        
        # Track su frame successivi
        for i in range(1, 5):
            success, centers = tracker.update(test_frames[i])
            
            assert success == True
            assert centers is not None
            assert len(centers) == 2
            
            # Verifica che i centri si siano mossi nella direzione corretta
            assert centers[0][0] > initial_points[0][0]
            assert centers[1][0] > initial_points[1][0]
    
    def test_tracking_loss(self, tracking_config):
        """Test comportamento quando tracking è perso."""
        tracker = LightTracker(tracking_config)
        
        # Frame iniziale
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame1, (200, 300), 20, (0, 0, 255), -1)
        cv2.circle(frame1, (440, 300), 20, (0, 0, 255), -1)
        
        # Frame vuoto (tracking perso)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        initial_points = ((200, 300), (440, 300))
        tracker.initialize(frame1, initial_points)
        
        # Update su frame vuoto
        success, centers = tracker.update(frame2)
        
        assert success == False
        assert tracker.frames_since_detection > 0
    
    def test_needs_redetection(self, tracking_config):
        """Test logica per redetection."""
        tracker = LightTracker(tracking_config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.initialize(frame, ((200, 300), (440, 300)))
        
        # Simula frames persi
        tracker.frames_since_detection = 5
        assert tracker.needs_redetection() == False
        
        tracker.frames_since_detection = 15
        assert tracker.needs_redetection() == True
    
    def test_reinitialize(self, tracking_config, test_frames):
        """Test reinizializzazione tracker."""
        tracker = LightTracker(tracking_config)
        
        tracker.initialize(test_frames[0], ((200, 300), (440, 300)))
        
        # Reinizializza con nuovi punti
        new_points = ((250, 320), (490, 320))
        tracker.reinitialize(test_frames[5], new_points)
        
        assert tracker.is_initialized == True
        assert tracker.current_centers == new_points
        assert tracker.frames_since_detection == 0
    
    def test_reset(self, tracking_config, test_frames):
        """Test reset completo tracker."""
        tracker = LightTracker(tracking_config)
        
        tracker.initialize(test_frames[0], ((200, 300), (440, 300)))
        assert tracker.is_initialized == True
        
        tracker.reset()
        
        assert tracker.is_initialized == False
        assert len(tracker.trackers) == 0
        assert tracker.current_centers is None
    
    def test_point_to_bbox_conversion(self, tracking_config):
        """Test conversione punto -> bbox."""
        tracker = LightTracker(tracking_config)
        
        point = (100, 200)
        bbox = tracker._point_to_bbox(point)
        
        assert len(bbox) == 4
        x, y, w, h = bbox
        
        # Bbox dovrebbe essere centrata sul punto
        assert x < 100
        assert y < 200
        assert w == 2 * tracker.bbox_padding
        assert h == 2 * tracker.bbox_padding
    
    def test_bbox_to_point_conversion(self, tracking_config):
        """Test conversione bbox -> punto."""
        tracker = LightTracker(tracking_config)
        
        bbox = (80, 180, 40, 40)
        point = tracker._bbox_to_point(bbox)
        
        assert point == (100, 200)


class TestMultiObjectTracker:
    """Test per MultiObjectTracker (opzionale)."""
    
    def test_register_object(self):
        """Test registrazione oggetto."""
        from src.tracking.tracker import MultiObjectTracker
        
        tracker = MultiObjectTracker()
        
        obj_id = tracker.register((100, 200))
        
        assert obj_id == 0
        assert obj_id in tracker.objects
        assert tracker.objects[obj_id] == (100, 200)
    
    def test_update_with_detections(self):
        """Test update con nuove detection."""
        from src.tracking.tracker import MultiObjectTracker
        
        tracker = MultiObjectTracker()
        
        # Prima detection
        detections1 = [(100, 200), (300, 400)]
        objects = tracker.update(detections1)
        
        assert len(objects) == 2
        
        # Detection successive (oggetti si muovono leggermente)
        detections2 = [(105, 205), (305, 405)]
        objects = tracker.update(detections2)
        
        assert len(objects) == 2
    
    def test_object_disappearance(self):
        """Test rimozione oggetti scomparsi."""
        from src.tracking.tracker import MultiObjectTracker
        
        tracker = MultiObjectTracker(max_disappeared=2)
        
        # Registra oggetto
        tracker.register((100, 200))
        
        # Update senza detection per 3 volte
        for _ in range(3):
            tracker.update([])
        
        # Oggetto dovrebbe essere stato rimosso
        assert len(tracker.objects) == 0


def test_tracking_integration(tracking_config, test_frames):
    """Test integrazione completa tracking."""
    tracker = LightTracker(tracking_config)
    
    initial_points = ((200, 300), (440, 300))
    tracker.initialize(test_frames[0], initial_points)
    
    all_centers = [initial_points]
    
    # Track attraverso tutti i frame
    for i in range(1, len(test_frames)):
        success, centers = tracker.update(test_frames[i])
        
        if success:
            all_centers.append(centers)
    
    # Dovrebbe aver tracciato la maggior parte dei frame
    assert len(all_centers) >= len(test_frames) * 0.8
    
    # Verifica movimento progressivo
    for i in range(1, len(all_centers)):
        # X dovrebbe aumentare
        assert all_centers[i][0][0] >= all_centers[i-1][0][0]
        assert all_centers[i][1][0] >= all_centers[i-1][1][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])