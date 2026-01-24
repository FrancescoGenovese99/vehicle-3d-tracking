"""
Kalman Filter Tracker per luci posteriori
Predice posizione quando detection fallisce
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class KalmanLightTracker:
    """
    Tracker Kalman 2D per una singola luce.
    
    State: [x, y, vx, vy] (posizione + velocità)
    Measurement: [x, y]
    """
    
    def __init__(self, initial_position: np.ndarray):
        """
        Args:
            initial_position: [x, y] posizione iniziale
        """
        # Kalman Filter (4 stati, 2 misure)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix: x_k = A * x_{k-1}
        # [x]   [1 0 dt 0 ] [x  ]
        # [y] = [0 1 0  dt] [y  ]
        # [vx]  [0 0 1  0 ] [vx ]
        # [vy]  [0 0 0  1 ] [vy ]
        dt = 1.0  # Assume 1 frame = 1 unità tempo
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)
        
        # Measurement matrix: z = H * x
        # [x] = [1 0 0 0] [x ]
        # [y]   [0 1 0 0] [y ]
        #                 [vx]
        #                 [vy]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initialize state
        self.kf.statePost = np.array([
            initial_position[0],
            initial_position[1],
            0.0,  # vx
            0.0   # vy
        ], dtype=np.float32)
        
        self.lost_frames = 0
        self.max_lost_frames = 10
    
    def predict(self) -> np.ndarray:
        """
        Predice posizione prossimo frame.
        
        Returns:
            [x, y] predetto
        """
        prediction = self.kf.predict()
        return np.array([prediction[0], prediction[1]], dtype=np.float32)
    
    def update(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        Aggiorna con misura (se disponibile).
        
        Args:
            measurement: [x, y] misurato, o None se detection fallita
            
        Returns:
            [x, y] stimato corrente
        """
        if measurement is not None:
            # Misura disponibile: update
            self.kf.correct(measurement.astype(np.float32))
            self.lost_frames = 0
            
            state = self.kf.statePost
            return np.array([state[0], state[1]], dtype=np.float32)
        else:
            # Misura mancante: usa predizione
            self.lost_frames += 1
            
            prediction = self.predict()
            return prediction
    
    def is_lost(self) -> bool:
        """Verifica se tracker è perso (troppi frame senza misure)."""
        return self.lost_frames > self.max_lost_frames


class DualLightTracker:
    """Tracker per coppia di luci (left + right)."""
    
    def __init__(self, initial_lights: np.ndarray):
        """
        Args:
            initial_lights: [[L_x, L_y], [R_x, R_y]]
        """
        self.tracker_left = KalmanLightTracker(initial_lights[0])
        self.tracker_right = KalmanLightTracker(initial_lights[1])
    
    def predict(self) -> np.ndarray:
        """Predice entrambe le luci."""
        L_pred = self.tracker_left.predict()
        R_pred = self.tracker_right.predict()
        return np.array([L_pred, R_pred], dtype=np.float32)
    
    def update(
        self,
        measurements: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, bool]:
        """
        Aggiorna tracker.
        
        Args:
            measurements: [[L_x, L_y], [R_x, R_y]] o None
            
        Returns:
            (positions, is_measurement_available)
        """
        if measurements is not None:
            L_meas = measurements[0]
            R_meas = measurements[1]
            
            L_est = self.tracker_left.update(L_meas)
            R_est = self.tracker_right.update(R_meas)
            
            return np.array([L_est, R_est], dtype=np.float32), True
        else:
            # Usa predizioni
            L_est = self.tracker_left.update(None)
            R_est = self.tracker_right.update(None)
            
            return np.array([L_est, R_est], dtype=np.float32), False
    
    def is_lost(self) -> bool:
        """Verifica se entrambi i tracker sono persi."""
        return self.tracker_left.is_lost() and self.tracker_right.is_lost()
    
    def reset(self, new_lights: np.ndarray):
        """Reset tracker con nuove posizioni."""
        self.tracker_left = KalmanLightTracker(new_lights[0])
        self.tracker_right = KalmanLightTracker(new_lights[1])