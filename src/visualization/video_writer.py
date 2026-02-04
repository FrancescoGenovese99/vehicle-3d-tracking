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
        Inizializza il video writer con fallback automatico su codec.
        
        Args:
            output_path: Path del file di output
            fps: Frame rate del video
            frame_size: Dimensioni frame (width, height). Se None, sarà impostato dal primo frame
            codec: Codec fourcc ('mp4v', 'XVID', 'MJPG', ecc.)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        # Crea directory se non esiste
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # VideoWriter
        self.writer = None
        self.is_initialized = False
        self.frame_count = 0
        
        # IMPORTANTE: Se frame_size è fornito, inizializza SUBITO
        if frame_size is not None:
            self._initialize_writer(frame_size)
    
    def _initialize_writer(self, frame_size: Tuple[int, int]):
        """
        Inizializza il VideoWriter con fallback automatico su codec.
        
        Args:
            frame_size: Dimensioni frame (width, height)
        """
        # Lista codec da provare in ordine di preferenza
        codecs_to_try = [
            self.codec,     # Codec richiesto dall'utente
            'avc1',         # H.264 (molto compatibile)
            'XVID',         # Fallback 1
            'MJPG',         # Fallback 2 (quasi sempre funziona)
        ]
        
        # Rimuovi duplicati mantenendo l'ordine
        seen = set()
        codecs_to_try = [x for x in codecs_to_try if not (x in seen or seen.add(x))]
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    frame_size
                )
                
                if self.writer.isOpened():
                    self.is_initialized = True
                    print(f"✓ VideoWriter inizializzato: {self.output_path}")
                    print(f"  FPS: {self.fps}, Size: {frame_size}, Codec: {codec}")
                    return  # Successo!
                else:
                    # Fallito, rilascia e prova prossimo
                    if self.writer is not None:
                        self.writer.release()
                    
            except Exception as e:
                # Errore durante creazione fourcc o writer
                continue
        
        # Se arriviamo qui, nessun codec ha funzionato
        raise RuntimeError(
            f"Impossibile aprire VideoWriter per {self.output_path}\n"
            f"Codec provati: {codecs_to_try}\n"
            f"Frame size: {frame_size}, FPS: {self.fps}"
        )
    
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
    
    def write(self, frame: np.ndarray):
        """Alias per write_frame() - compatibilità con OpenCV VideoWriter."""
        self.write_frame(frame)
    
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


# ============================================================================
# ALIAS PER COMPATIBILITÀ - USA DIRETTAMENTE VideoWriterManager
# ============================================================================
VideoWriter = VideoWriterManager