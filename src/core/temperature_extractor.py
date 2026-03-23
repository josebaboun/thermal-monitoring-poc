"""Temperature extraction from thermal video frames."""
import cv2
import numpy as np
from typing import Optional


# HIKMICRO PocketE overlay regions (in a 240x240 frame)
# Top: ~35px of text overlay (Cen, Max, Min, HIKMICRO logo)
# Bottom: ~15px of status bar
# Left: ~8px of color scale bar
OVERLAY_TOP = 35
OVERLAY_BOTTOM = 15
OVERLAY_LEFT = 8


class TemperatureExtractor:
    """Extracts temperature data from thermal video frames."""

    def __init__(self, config: dict):
        self.temp_min = config.get('temp_range_min', 20)
        self.temp_max = config.get('temp_range_max', 300)

    def extract_from_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract approximate temperature from RGB thermal frame.

        Uses grayscale intensity mapped linearly to the configured
        temperature range. The overlay regions are masked to avoid
        interference from the camera's on-screen text/scale.
        """
        h, w = frame_rgb.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Simple linear mapping: 0 (darkest) → temp_min, 255 (brightest) → temp_max
        temp_celsius = (gray / 255.0) * (self.temp_max - self.temp_min) + self.temp_min

        # Set overlay regions to temp_min so they don't trigger detections
        temp_celsius[:OVERLAY_TOP, :] = self.temp_min       # Top text
        temp_celsius[-OVERLAY_BOTTOM:, :] = self.temp_min   # Bottom bar
        temp_celsius[:, :OVERLAY_LEFT] = self.temp_min      # Left scale

        return temp_celsius

    def get_analysis_mask(self, height: int, width: int) -> np.ndarray:
        """Return a mask of the analysis region (excluding overlay)."""
        mask = np.ones((height, width), dtype=np.uint8) * 255
        mask[:OVERLAY_TOP, :] = 0
        mask[-OVERLAY_BOTTOM:, :] = 0
        mask[:, :OVERLAY_LEFT] = 0
        return mask
