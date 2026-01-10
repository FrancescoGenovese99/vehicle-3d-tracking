"""
Test per i moduli di detection (light_detector e candidate_selector).
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.light_detector import LightDetector, LightCandidate
from src.detection.candidate_selector import CandidateSelector
from src.utils.config_loader import load_config


@pytest.fixture
def detection_config():
    """Carica configurazione di detection."""
    config_path = Path(__file__).parent.parent / "config" / "detection_params.yaml"
    if config_path.exists():
        return load_config(str(config_path))
    else:
        # Config di default per test
        return {
            'hsv_ranges': {
                'red': {
                    'lower1': [0, 100, 100],
                    'upper1': [10, 255, 255],
                    'lower2': [170, 100, 100],
                    'upper2': [180, 255, 255]
                },
                'white': {
                    'lower': [0, 0, 200],
                    'upper': [180, 30, 255]
                }
            },
            'blob_detection': {
                'min_area': 50,
                'max_area': 5000,
                'min_circularity': 0.4
            },
            'morphology': {
                'kernel_size': [5, 5],
                'open_iterations': 1,
                'close_iterations': 1
            },
            'tail_lights_selection': {
                'min_horizontal_distance': 50,
                'max_horizontal_distance_ratio': 0.8,
                'max_vertical_offset': 50,
                'min_area_similarity': 0.5,
                'min_pair_score': 0.3
            }
        }


@pytest.fixture
def synthetic_frame_with_lights():
    """Crea un frame sintetico con due fari rossi."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Faro sinistro (rosso)
    cv2.circle(frame, (200, 300), 30, (0, 0, 255), -1)
    
    # Faro destro (rosso)
    cv2.circle(frame, (440, 300), 30, (0, 0, 255), -1)
    
    return frame


class TestLightDetector:
    """Test per LightDetector."""
    
    def test_initialization(self, detection_config):
        """Test inizializzazione detector."""
        detector = LightDetector(detection_config)
        
        assert detector is not None
        assert detector.min_area == 50
        assert detector.max_area == 5000
    
    def test_detect_red_lights(self, detection_config, synthetic_frame_with_lights):
        """Test detection di luci rosse."""
        detector = LightDetector(detection_config)
        mask = detector.detect_red_lights(synthetic_frame_with_lights)
        
        assert mask is not None
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)  # Deve rilevare qualcosa
    
    def test_detect_white_lights(self, detection_config):
        """Test detection di luci bianche."""
        detector = LightDetector(detection_config)
        
        # Frame con luci bianche
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 20, (255, 255, 255), -1)
        
        mask = detector.detect_white_lights(frame)
        
        assert mask is not None
        assert np.any(mask > 0)
    
    def test_find_light_candidates(self, detection_config, synthetic_frame_with_lights):
        """Test ricerca candidati."""
        detector = LightDetector(detection_config)
        candidates, mask = detector.detect_tail_lights(synthetic_frame_with_lights)
        
        assert candidates is not None
        assert len(candidates) >= 2  # Dovrebbe trovare almeno 2 fari
        
        for candidate in candidates:
            assert isinstance(candidate, LightCandidate)
            assert candidate.center is not None
            assert candidate.area > 0
    
    def test_morphology(self, detection_config):
        """Test operazioni morfologiche."""
        detector = LightDetector(detection_config)
        
        # Mask con rumore
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        mask[10, 10] = 255  # Pixel isolato
        
        cleaned = detector.apply_morphology(mask)
        
        assert cleaned is not None
        assert cleaned[10, 10] == 0  # Il rumore dovrebbe essere rimosso


class TestCandidateSelector:
    """Test per CandidateSelector."""
    
    def test_initialization(self, detection_config):
        """Test inizializzazione selector."""
        selector = CandidateSelector(detection_config, frame_width=640)
        
        assert selector is not None
        assert selector.frame_width == 640
    
    def test_compute_pair_score(self, detection_config):
        """Test calcolo score di coppia."""
        selector = CandidateSelector(detection_config, frame_width=640)
        
        # Due candidati validi
        c1 = LightCandidate(
            center=(200, 300),
            contour=np.array([]),
            area=500,
            circularity=0.8,
            bbox=(190, 290, 20, 20)
        )
        
        c2 = LightCandidate(
            center=(440, 300),
            contour=np.array([]),
            area=480,
            circularity=0.75,
            bbox=(430, 290, 20, 20)
        )
        
        score, metrics = selector.compute_pair_score(c1, c2)
        
        assert 0 <= score <= 1
        assert 'dx' in metrics
        assert 'dy' in metrics
        assert metrics['dx'] == 240
        assert metrics['dy'] == 0
    
    def test_select_tail_light_pair(self, detection_config):
        """Test selezione coppia fari."""
        selector = CandidateSelector(detection_config, frame_width=640)
        
        # Crea candidati
        candidates = [
            LightCandidate((200, 300), np.array([]), 500, 0.8, (0, 0, 20, 20)),
            LightCandidate((440, 300), np.array([]), 480, 0.75, (0, 0, 20, 20)),
            LightCandidate((100, 400), np.array([]), 300, 0.6, (0, 0, 15, 15))  # Outlier
        ]
        
        pair = selector.select_tail_light_pair(candidates)
        
        assert pair is not None
        assert len(pair) == 2
        assert pair[0].center == (200, 300)
        assert pair[1].center == (440, 300)
    
    def test_get_tail_light_centers(self, detection_config):
        """Test estrazione centri."""
        selector = CandidateSelector(detection_config, frame_width=640)
        
        candidates = [
            LightCandidate((200, 300), np.array([]), 500, 0.8, (0, 0, 20, 20)),
            LightCandidate((440, 300), np.array([]), 480, 0.75, (0, 0, 20, 20))
        ]
        
        centers = selector.get_tail_light_centers(candidates)
        
        assert centers is not None
        assert len(centers) == 2
        assert centers[0] == (200, 300)
        assert centers[1] == (440, 300)
    
    def test_filter_by_previous_position(self, detection_config):
        """Test filtro per posizione precedente."""
        selector = CandidateSelector(detection_config, frame_width=640)
        
        previous_centers = ((200, 300), (440, 300))
        
        candidates = [
            LightCandidate((205, 305), np.array([]), 500, 0.8, (0, 0, 20, 20)),  # Vicino
            LightCandidate((445, 295), np.array([]), 480, 0.75, (0, 0, 20, 20)),  # Vicino
            LightCandidate((100, 100), np.array([]), 300, 0.6, (0, 0, 15, 15))   # Lontano
        ]
        
        filtered = selector.filter_by_previous_position(
            candidates, previous_centers, max_distance=50
        )
        
        assert len(filtered) == 2
        assert all(c.center != (100, 100) for c in filtered)


def test_integration_detection_and_selection(detection_config, synthetic_frame_with_lights):
    """Test integrazione completa detection + selection."""
    detector = LightDetector(detection_config)
    selector = CandidateSelector(detection_config, frame_width=640)
    
    # Detection
    candidates, mask = detector.detect_tail_lights(synthetic_frame_with_lights)
    
    # Selection
    centers = selector.get_tail_light_centers(candidates)
    
    assert centers is not None
    assert len(centers) == 2
    
    # Verifica che i centri siano vicini a quelli attesi
    expected_left = (200, 300)
    expected_right = (440, 300)
    
    tolerance = 50  # pixel
    
    left_match = any(
        abs(c[0] - expected_left[0]) < tolerance and 
        abs(c[1] - expected_left[1]) < tolerance 
        for c in centers
    )
    
    right_match = any(
        abs(c[0] - expected_right[0]) < tolerance and 
        abs(c[1] - expected_right[1]) < tolerance 
        for c in centers
    )
    
    assert left_match and right_match


if __name__ == '__main__':
    pytest.main([__file__, '-v'])