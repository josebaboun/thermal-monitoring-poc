"""Video processing utilities for thermal video."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoProcessor:
    """Handles loading and processing of thermal video files."""

    def __init__(self, config: dict):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 0
        self.width: int = 0
        self.height: int = 0
        self.current_frame: int = 0

    def load(self, video_path: str) -> bool:
        """Load video file and extract metadata."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Extract video metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def get_video_info(self) -> dict:
        """Get video metadata."""
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
        }

    def get_timestamp(self) -> str:
        """Get current timestamp in video."""
        if self.fps > 0:
            seconds = self.current_frame / self.fps
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            millisecs = int((seconds % 1) * 1000)
            return f"{minutes:02d}:{secs:02d}.{millisecs:03d}"
        return "00:00.000"

    def create_writer(self, output_path: str, fps: Optional[float] = None) -> cv2.VideoWriter:
        """Create video writer for output."""
        if fps is None:
            fps = self.fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.width, self.height)
        )

        if not writer.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")

        return writer

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
