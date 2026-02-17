"""
video_writer.py - Thin wrapper around cv2.VideoWriter with automatic codec fallback.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoWriterManager:
    """
    Manages video output via cv2.VideoWriter.

    If the requested codec is unavailable on the current system, the class
    automatically tries a list of fallback codecs (avc1 → XVID → MJPG) so
    the pipeline keeps running without manual intervention.

    Supports the context-manager protocol for safe resource cleanup::

        with VideoWriterManager("out.mp4", fps=30) as writer:
            writer.write(frame)
    """

    # Codecs tried in order when the requested one fails
    _CODEC_FALLBACKS = ['avc1', 'XVID', 'MJPG']

    def __init__(self,
                 output_path: str,
                 fps: float = 30.0,
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: str = 'mp4v'):
        """
        Args:
            output_path: Destination file path (parent directories are created automatically).
            fps:         Output frame rate.
            frame_size:  (width, height) in pixels. If None, inferred from the first frame.
            codec:       Preferred FourCC codec string.
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer: Optional[cv2.VideoWriter] = None
        self.is_initialized = False
        self.frame_count = 0

        if frame_size is not None:
            self._initialize_writer(frame_size)

    def _initialize_writer(self, frame_size: Tuple[int, int]):
        """
        Tries to open a VideoWriter with the preferred codec, then falls back
        to alternatives until one succeeds.

        Raises:
            RuntimeError: If no codec could be opened.
        """
        codecs_to_try = [self.codec] + [c for c in self._CODEC_FALLBACKS if c != self.codec]

        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, frame_size)
                if writer.isOpened():
                    self.writer = writer
                    self.is_initialized = True
                    print(f"VideoWriter ready: {self.output_path}  "
                          f"[codec={codec}, fps={self.fps}, size={frame_size}]")
                    return
                writer.release()
            except Exception:
                continue

        raise RuntimeError(
            f"Could not open VideoWriter for '{self.output_path}'. "
            f"Tried codecs: {codecs_to_try}  frame_size={frame_size}  fps={self.fps}"
        )

    def write_frame(self, frame: np.ndarray):
        """
        Writes a BGR frame to the output video.
        Initializes the writer automatically on the first call if frame_size was not
        provided at construction time.
        """
        if frame is None:
            return
        if not self.is_initialized:
            h, w = frame.shape[:2]
            self._initialize_writer((w, h))
        self.writer.write(frame)
        self.frame_count += 1

    def write(self, frame: np.ndarray):
        """Alias for write_frame() — matches the cv2.VideoWriter interface."""
        self.write_frame(frame)

    def release(self):
        """Flushes and closes the output file."""
        if self.writer is not None and self.is_initialized:
            self.writer.release()
            print(f"Video saved: {self.output_path}  ({self.frame_count} frames)")
            self.is_initialized = False

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()


# Alias kept for backward compatibility
VideoWriter = VideoWriterManager