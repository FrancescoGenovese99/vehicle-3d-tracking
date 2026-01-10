"""
Test per i moduli di pose estimation.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_estimation.pnp_solver import PnPSolver
from src.pose_estimation.bbox_3d_projector import BBox3DProjector
from src.calibration.load_calibration import CameraParameters


@pytest.fixture
def camera_params():
    """Parametri camera di test."""
    camera_matrix = np.array([
        [800.0, 0, 320.0],
        [0, 800.0, 240.0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros(5, dtype=np.float64)
    
    return CameraParameters(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        resolution=(640, 480),
        fps=30.0
    )


@pytest.fixture
def vehicle_config():
    """Configurazione veicolo di test."""
    return {
        'vehicle': {
            'dimensions': {
                'length': 5.0,
                'width': 3.0,
                'height': 1.7
            },
            'tail_lights': {
                'left': [-1.2, 0.7, 0.5],
                'right': [-1.2, -0.7, 0.5]
            },
            'license_plate_rear': {
                'center': [0.0, 0.0, 0.4]
            }
        }
    }


@pytest.fixture
def pnp_config():
    """Configurazione PnP di test."""
    return {
        'method': 'ITERATIVE',
        'use_extrinsic_guess': False,
        'refine_iterations': 10,
        'ransac': {
            'enabled': False,
            'reprojection_error': 8.0,
            'confidence': 0.99
        }
    }


class TestPnPSolver:
    """Test per PnPSolver."""
    
    def test_initialization(self, camera_params, vehicle_config, pnp_config):
        """Test inizializzazione solver."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        assert solver is not None
        assert solver.camera_matrix is not None
        assert solver.object_points_3d.shape[0] >= 2  # Almeno 2 fari
    
    def test_build_object_points(self, camera_params, vehicle_config, pnp_config):
        """Test costruzione punti 3D oggetto."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        points = solver.object_points_3d
        
        assert points.shape[1] == 3  # Coordinate 3D
        assert len(points) >= 2  # Almeno i 2 fari
        
        # Verifica che i punti corrispondano alla config
        expected_left = np.array([-1.2, 0.7, 0.5])
        expected_right = np.array([-1.2, -0.7, 0.5])
        
        np.testing.assert_array_almost_equal(points[0], expected_left)
        np.testing.assert_array_almost_equal(points[1], expected_right)
    
    def test_solve_pnp_valid_points(self, camera_params, vehicle_config, pnp_config):
        """Test soluzione PnP con punti validi."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        # Crea punti immagine simulati
        # Simula un veicolo a 10m di distanza
        object_points = solver.object_points_3d
        
        # Posa ground truth
        rvec_true = np.array([[0.0], [0.0], [0.0]])  # Nessuna rotazione
        tvec_true = np.array([[0.0], [0.0], [10.0]])  # 10m avanti
        
        # Proietta per ottenere punti immagine
        image_points, _ = cv2.projectPoints(
            object_points,
            rvec_true,
            tvec_true,
            solver.camera_matrix,
            solver.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2)
        
        # Risolvi PnP
        success, rvec, tvec = solver.solve(image_points)
        
        assert success == True
        assert rvec is not None
        assert tvec is not None
        assert rvec.shape == (3, 1)
        assert tvec.shape == (3, 1)
        
        # Verifica che sia vicino alla ground truth
        np.testing.assert_array_almost_equal(rvec, rvec_true, decimal=1)
        np.testing.assert_array_almost_equal(tvec, tvec_true, decimal=0)
    
    def test_solve_pnp_insufficient_points(self, camera_params, vehicle_config, pnp_config):
        """Test con numero insufficiente di punti."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        # Solo 1 punto (insufficiente)
        image_points = np.array([[320.0, 240.0]], dtype=np.float32)
        
        success, rvec, tvec = solver.solve(image_points)
        
        assert success == False
    
    def test_compute_reprojection_error(self, camera_params, vehicle_config, pnp_config):
        """Test calcolo errore di riproiezione."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        # Posa di test
        rvec = np.array([[0.0], [0.0], [0.0]])
        tvec = np.array([[0.0], [0.0], [10.0]])
        
        # Proietta punti
        image_points, _ = cv2.projectPoints(
            solver.object_points_3d,
            rvec,
            tvec,
            solver.camera_matrix,
            solver.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2)
        
        # Calcola errore (dovrebbe essere ~0)
        error = solver.compute_reprojection_error(image_points, rvec, tvec)
        
        assert error < 0.01  # Errore molto piccolo
    
    def test_rvec_to_rotation_matrix(self, camera_params, vehicle_config, pnp_config):
        """Test conversione Rodrigues -> matrice rotazione."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        rvec = np.array([[0.0], [0.0], [0.0]])  # Identità
        R = solver.rvec_to_rotation_matrix(rvec)
        
        assert R.shape == (3, 3)
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_rotation_matrix_to_euler(self, camera_params, vehicle_config, pnp_config):
        """Test conversione matrice rotazione -> angoli Eulero."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        R = np.eye(3)  # Identità
        roll, pitch, yaw = solver.rotation_matrix_to_euler(R)
        
        assert abs(roll) < 0.01
        assert abs(pitch) < 0.01
        assert abs(yaw) < 0.01
    
    def test_get_pose_info(self, camera_params, vehicle_config, pnp_config):
        """Test estrazione info posa."""
        solver = PnPSolver(camera_params, vehicle_config, pnp_config)
        
        rvec = np.array([[0.0], [0.0], [0.0]])
        tvec = np.array([[1.0], [2.0], [10.0]])
        
        info = solver.get_pose_info(rvec, tvec)
        
        assert 'translation' in info
        assert 'rotation' in info
        assert info['translation']['x'] == 1.0
        assert info['translation']['y'] == 2.0
        assert info['translation']['z'] == 10.0
        assert 'distance' in info['translation']


class TestBBox3DProjector:
    """Test per BBox3DProjector."""
    
    def test_initialization(self, camera_params, vehicle_config):
        """Test inizializzazione projector."""
        projector = BBox3DProjector(camera_params, vehicle_config)
        
        assert projector is not None
        assert projector.length == 5.0
        assert projector.width == 3.0
        assert projector.height == 1.7
    
    def test_compute_bbox_vertices(self, camera_params, vehicle_config):
        """Test calcolo vertici bbox."""
        projector = BBox3DProjector(camera_params, vehicle_config)
        
        vertices = projector.bbox_vertices_3d
        
        assert vertices.shape == (8, 3)  # 8 vertici, 3D
        
        # Verifica che ci siano 4 vertici a z=0 e 4 a z=height
        z_values = vertices[:, 2]
        assert np.sum(z_values == 0) == 4
        assert np.sum(z_values == projector.height) == 4
    
    def test_project_bbox(self, camera_params, vehicle_config):
        """Test proiezione bbox."""
        projector = BBox3DProjector(camera_params, vehicle_config)
        
        # Posa di test
        rvec = np.array([[0.0], [0.0], [0.0]])
        tvec = np.array([[0.0], [0.0], [10.0]])
        
        projected = projector.project_bbox(rvec, tvec)
        
        assert projected.shape == (8, 2)  # 8 punti 2D
        
        # Verifica che i punti siano nel range ragionevole
        assert np.all(projected[:, 0] >= 0)
        assert np.all(projected[:, 0] < 640)
        assert np.all(projected[:, 1] >= 0)
        assert np.all(projected[:, 1] < 480)
    
    def test_is_bbox_visible(self, camera_params, vehicle_config):
        """Test visibilità bbox."""
        projector = BBox3DProjector(camera_params, vehicle_config)
        
        # Punti visibili
        visible_points = np.array([
            [100, 100],
            [200, 200],
            [300, 300],
            [400, 400],
            [150, 150],
            [250, 250],
            [350, 350],
            [450, 450]
        ])
        
        assert projector.is_bbox_visible(visible_points, (480, 640)) == True
        
        # Punti fuori frame
        invisible_points = np.array([
            [700, 700],
            [800, 800],
            [900, 900],
            [1000, 1000],
            [750, 750],
            [850, 850],
            [950, 950],
            [1050, 1050]
        ])
        
        assert projector.is_bbox_visible(invisible_points, (480, 640)) == False
    
    def test_compute_bbox_area_2d(self, camera_params, vehicle_config):
        """Test calcolo area 2D bbox."""
        projector = BBox3DProjector(camera_params, vehicle_config)
        
        # Punti che formano un quadrato
        square_points = np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200],
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200]
        ])
        
        area = projector.compute_bbox_area_2d(square_points)
        
        assert area > 0
        assert area <= 10000  # Max area per un quadrato 100x100


def test_integration_pnp_and_projection(camera_params, vehicle_config, pnp_config):
    """Test integrazione PnP + proiezione bbox."""
    solver = PnPSolver(camera_params, vehicle_config, pnp_config)
    projector = BBox3DProjector(camera_params, vehicle_config)
    
    # Posa ground truth
    rvec_true = np.array([[0.0], [0.1], [0.0]])
    tvec_true = np.array([[0.0], [0.0], [10.0]])
    
    # Proietta punti oggetto
    image_points, _ = cv2.projectPoints(
        solver.object_points_3d,
        rvec_true,
        tvec_true,
        solver.camera_matrix,
        solver.dist_coeffs
    )
    
    image_points = image_points.reshape(-1, 2)
    
    # Risolvi PnP
    success, rvec, tvec = solver.solve(image_points)
    
    assert success == True
    
    # Proietta bbox con posa stimata
    projected_bbox = projector.project_bbox(rvec, tvec)
    
    assert projected_bbox is not None
    assert projected_bbox.shape == (8, 2)
    
    # Verifica che bbox sia visibile
    assert projector.is_bbox_visible(projected_bbox, (480, 640)) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])