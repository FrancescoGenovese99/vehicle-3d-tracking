"""
Draw Utils - Funzioni di utilità per disegnare su frame.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


class DrawUtils:
    """
    Utilità per disegnare elementi sui frame (tracking, bbox, info).
    """
    
    @staticmethod
    def draw_tracked_points(frame: np.ndarray,
                           points: Tuple[Tuple[int, int], Tuple[int, int]],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           radius: int = 5,
                           thickness: int = -1,
                           labels: bool = True) -> np.ndarray:
        """
        Disegna i punti tracciati (fari) sul frame.
        
        Args:
            frame: Frame su cui disegnare
            points: Tuple ((left_x, left_y), (right_x, right_y))
            color: Colore BGR
            radius: Raggio dei cerchi
            thickness: Spessore (-1 = riempito)
            labels: Se True, aggiunge etichette "L" e "R"
            
        Returns:
            Frame con punti disegnati
        """
        frame_copy = frame.copy()
        
        left, right = points
        
        # Disegna cerchi
        cv2.circle(frame_copy, left, radius, color, thickness)
        cv2.circle(frame_copy, right, radius, color, thickness)
        
        # Linea di connessione
        cv2.line(frame_copy, left, right, color, 2)
        
        if labels:
            # Etichette
            cv2.putText(frame_copy, 'L', (left[0] - 20, left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_copy, 'R', (right[0] + 10, right[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    @staticmethod
    def draw_pose_info(frame: np.ndarray,
                      pose_info: Dict,
                      position: Tuple[int, int] = (10, 30),
                      font_scale: float = 0.6,
                      color: Tuple[int, int, int] = (255, 255, 255),
                      thickness: int = 2,
                      bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                      bg_alpha: float = 0.7) -> np.ndarray:
        """
        Disegna informazioni sulla posa sul frame.
        
        Args:
            frame: Frame su cui disegnare
            pose_info: Dizionario con info posa (da PnPSolver.get_pose_info)
            position: Posizione iniziale del testo (x, y)
            font_scale: Scala del font
            color: Colore testo BGR
            thickness: Spessore testo
            bg_color: Colore sfondo (None = nessuno sfondo)
            bg_alpha: Trasparenza sfondo (0-1)
            
        Returns:
            Frame con info disegnate
        """
        frame_copy = frame.copy()
        
        # Estrai info
        trans = pose_info['translation']
        rot = pose_info['rotation']
        
        # Prepara testo
        lines = [
            f"Distance: {trans['distance']:.2f}m",
            f"Position: ({trans['x']:.2f}, {trans['y']:.2f}, {trans['z']:.2f})",
            f"Yaw: {rot['yaw']:.1f}deg",
            f"Pitch: {rot['pitch']:.1f}deg",
            f"Roll: {rot['roll']:.1f}deg"
        ]
        
        # Calcola dimensioni sfondo
        line_height = int(30 * font_scale)
        max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                         font_scale, thickness)[0][0] for line in lines])
        
        if bg_color:
            # Disegna sfondo semi-trasparente
            x, y = position
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, 
                         (x - 5, y - 20),
                         (x + max_width + 10, y + line_height * len(lines) + 10),
                         bg_color, -1)
            frame_copy = cv2.addWeighted(frame_copy, 1 - bg_alpha, overlay, bg_alpha, 0)
        
        # Disegna testo
        x, y = position
        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            cv2.putText(frame_copy, line, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return frame_copy
    
    @staticmethod
    def draw_tracking_status(frame: np.ndarray,
                           status: str,
                           frame_number: int,
                           position: Tuple[int, int] = None,
                           color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Disegna lo status del tracking.
        
        Args:
            frame: Frame su cui disegnare
            status: Status string ("TRACKING", "LOST", "REDETECTING")
            frame_number: Numero del frame corrente
            position: Posizione (None = auto in alto a destra)
            color: Colore (None = auto basato su status)
            
        Returns:
            Frame con status disegnato
        """
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        
        # Colori di default basati su status
        if color is None:
            color_map = {
                'TRACKING': (0, 255, 0),     # Verde
                'LOST': (0, 0, 255),          # Rosso
                'REDETECTING': (0, 165, 255), # Arancione
                'INITIALIZING': (255, 255, 0) # Giallo
            }
            color = color_map.get(status.upper(), (255, 255, 255))
        
        # Posizione di default (alto a destra)
        if position is None:
            position = (w - 250, 30)
        
        # Testo
        text = f"Frame: {frame_number} | {status}"
        
        # Sfondo
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x, y = position
        cv2.rectangle(frame_copy, 
                     (x - 5, y - 25),
                     (x + text_size[0] + 5, y + 5),
                     (0, 0, 0), -1)
        
        # Testo
        cv2.putText(frame_copy, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame_copy
    
    @staticmethod
    def draw_detection_confidence(frame: np.ndarray,
                                  confidence: float,
                                  position: Tuple[int, int] = (10, 60)) -> np.ndarray:
        """
        Disegna una barra di confidenza per la detection.
        
        Args:
            frame: Frame su cui disegnare
            confidence: Valore di confidenza (0-1)
            position: Posizione della barra
            
        Returns:
            Frame con barra disegnata
        """
        frame_copy = frame.copy()
        
        bar_width = 200
        bar_height = 20
        x, y = position
        
        # Bordo
        cv2.rectangle(frame_copy, (x, y), (x + bar_width, y + bar_height), 
                     (255, 255, 255), 2)
        
        # Riempimento basato su confidenza
        fill_width = int(bar_width * confidence)
        
        # Colore basato su confidenza
        if confidence > 0.7:
            color = (0, 255, 0)  # Verde
        elif confidence > 0.4:
            color = (0, 165, 255)  # Arancione
        else:
            color = (0, 0, 255)  # Rosso
        
        cv2.rectangle(frame_copy, (x, y), (x + fill_width, y + bar_height),
                     color, -1)
        
        # Testo
        text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame_copy, text, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    @staticmethod
    def draw_trajectory(frame: np.ndarray,
                       trajectory: List[Tuple[int, int]],
                       color: Tuple[int, int, int] = (255, 0, 255),
                       thickness: int = 2,
                       max_points: int = 50) -> np.ndarray:
        """
        Disegna la traiettoria del veicolo.
        
        Args:
            frame: Frame su cui disegnare
            trajectory: Lista di posizioni [(x, y), ...]
            color: Colore BGR
            thickness: Spessore linea
            max_points: Numero massimo di punti da visualizzare
            
        Returns:
            Frame con traiettoria disegnata
        """
        frame_copy = frame.copy()
        
        if len(trajectory) < 2:
            return frame_copy
        
        # Limita ai max_points più recenti
        recent_trajectory = trajectory[-max_points:]
        
        # Disegna linee tra punti consecutivi
        for i in range(len(recent_trajectory) - 1):
            pt1 = recent_trajectory[i]
            pt2 = recent_trajectory[i + 1]
            
            # Alpha fade per punti più vecchi
            alpha = (i + 1) / len(recent_trajectory)
            line_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame_copy, pt1, pt2, line_color, thickness)
        
        # Disegna un cerchio sull'ultimo punto
        cv2.circle(frame_copy, recent_trajectory[-1], 5, color, -1)
        
        return frame_copy
    
    @staticmethod
    def create_split_view(frames: List[np.ndarray],
                         labels: Optional[List[str]] = None,
                         layout: str = 'horizontal') -> np.ndarray:
        """
        Crea una vista split con più frame affiancati.
        
        Args:
            frames: Lista di frame da affiancare
            labels: Etichette opzionali per ogni frame
            layout: 'horizontal' o 'vertical'
            
        Returns:
            Frame combinato
        """
        if not frames:
            return None
        
        # Assicurati che tutti i frame abbiano le stesse dimensioni
        h, w = frames[0].shape[:2]
        resized_frames = []
        
        for frame in frames:
            if frame.shape[:2] != (h, w):
                resized = cv2.resize(frame, (w, h))
            else:
                resized = frame.copy()
            
            resized_frames.append(resized)
        
        # Aggiungi etichette se fornite
        if labels:
            for i, (frame, label) in enumerate(zip(resized_frames, labels)):
                cv2.putText(frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Combina frame
        if layout == 'horizontal':
            combined = np.hstack(resized_frames)
        else:  # vertical
            combined = np.vstack(resized_frames)
        
        return combined