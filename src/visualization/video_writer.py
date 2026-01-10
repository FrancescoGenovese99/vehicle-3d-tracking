"""
Video Writer Manager - Gestione del salvataggio video.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoWriterManager:
    """
    Gestisce la scrittura di video con OpenCV VideoWriter.
    """
    
    def __init__(self, output_path: str, 
                 fps: float = 30.0,
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: str = 'mp4v'):
        """
        Inizializza il video writer.
        
        Args:
            output_path: Path del file di output
            fps: Frame rate del video
            frame_size: Dimensioni frame (width, height). Se None, sarà impostato dal primo frame
            codec: Codec fourcc ('mp4v', 'XVID', 'H264', ecc.)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        # Crea directory se non esiste
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # VideoWriter sarà inizializzato al primo frame se frame_size è None
        self.writer = None
        self.is_initialized = False
        self.frame_count = 0
    
    def _initialize_writer(self, frame_size: Tuple[int, int]):
        """
        Inizializza il VideoWriter.
        
        Args:
            frame_size: Dimensioni frame (width, height)
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Impossibile aprire VideoWriter per {self.output_path}")
        
        self.is_initialized = True
        print(f"VideoWriter inizializzato: {self.output_path}")
        print(f"  FPS: {self.fps}, Size: {frame_size}, Codec: {self.codec}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Scrive un frame sul video.
        
        Args:
            frame: Frame da scrivere (BGR)
        """
        if frame is None:
            return
        
        # Inizializza writer se necessario
        if not self.is_initialized:
            h, w = frame.shape[:2]
            self._initialize_writer((w, h))
        
        # Scrivi frame
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Rilascia il VideoWriter."""
        if self.writer is not None and self.is_initialized:
            self.writer.release()
            print(f"✓ Video salvato: {self.output_path}")
            print(f"  Frames scritti: {self.frame_count}")
            self.is_initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


class MultiVideoWriter:
    """
    Gestisce la scrittura di più video contemporaneamente (es. views diverse).
    """
    
    def __init__(self, output_paths: dict, fps: float = 30.0, codec: str = 'mp4v'):
        """
        Inizializza multi video writer.
        
        Args:
            output_paths: Dizionario {name: path} per ogni video
            fps: Frame rate
            codec: Codec fourcc
        """
        self.writers = {}
        
        for name, path in output_paths.items():
            self.writers[name] = VideoWriterManager(path, fps, codec=codec)
    
    def write_frames(self, frames: dict):
        """
        Scrive frame multipli.
        
        Args:
            frames: Dizionario {name: frame} corrispondente ai writer
        """
        for name, frame in frames.items():
            if name in self.writers:
                self.writers[name].write_frame(frame)
    
    def release(self):
        """Rilascia tutti i writer."""
        for writer in self.writers.values():
            writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def video_to_frames(video_path: str, output_dir: str, 
                   frame_prefix: str = "frame",
                   max_frames: Optional[int] = None) -> int:
    """
    Estrae frame da un video e li salva come immagini.
    
    Args:
        video_path: Path del video
        output_dir: Directory di output per i frame
        frame_prefix: Prefisso per i nomi dei file
        max_frames: Numero massimo di frame da estrarre (None = tutti)
        
    Returns:
        Numero di frame estratti
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire video: {video_path}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Salva frame
        output_file = output_path / f"{frame_prefix}_{frame_count:06d}.jpg"
        cv2.imwrite(str(output_file), frame)
        
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    print(f"✓ Estratti {frame_count} frame in {output_dir}")
    
    return frame_count


def frames_to_video(frames_pattern: str, output_path: str,
                   fps: float = 30.0, codec: str = 'mp4v'):
    """
    Crea un video da una sequenza di frame.
    
    Args:
        frames_pattern: Pattern glob per i frame (es: "frames/*.jpg")
        output_path: Path del video di output
        fps: Frame rate
        codec: Codec fourcc
    """
    import glob
    
    # Ottieni lista frame ordinata
    frame_files = sorted(glob.glob(frames_pattern))
    
    if not frame_files:
        raise ValueError(f"Nessun frame trovato con pattern: {frames_pattern}")
    
    # Leggi primo frame per ottenere dimensioni
    first_frame = cv2.imread(frame_files[0])
    h, w = first_frame.shape[:2]
    
    # Crea writer
    with VideoWriterManager(output_path, fps, (w, h), codec) as writer:
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                writer.write_frame(frame)
    
    print(f"✓ Video creato da {len(frame_files)} frame")