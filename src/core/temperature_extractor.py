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
        Extract approximate temperature from RGB thermal frame using
        HSV-based rainbow colormap decoding.

        In a rainbow colormap, hue maps inversely to temperature:
        - Blue (H~120 in OpenCV) = coldest
        - Cyan (H~90) = cool
        - Green (H~60) = medium
        - Yellow (H~30) = warm
        - Red (H~0) = hottest
        Low-saturation bright pixels (near white) = very hot.
        """
        h, w = frame_rgb.shape[:2]

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)       # 0-179
        sat = hsv[:, :, 1].astype(np.float32)       # 0-255
        val = hsv[:, :, 2].astype(np.float32)       # 0-255

        # For saturated pixels: temperature is inverse of hue (capped at 120)
        # Hue 120 (blue) = 0.0 (cold), Hue 0 (red) = 1.0 (hot)
        hue_clamped = np.clip(hue, 0, 120)
        normalized = 1.0 - (hue_clamped / 120.0)

        # For low-saturation bright pixels (near white): very hot
        white_mask = (sat < 60) & (val > 200)
        normalized[white_mask] = 1.0

        # For dark pixels (near black): cold / background
        dark_mask = val < 30
        normalized[dark_mask] = 0.0

        temp_celsius = normalized * (self.temp_max - self.temp_min) + self.temp_min

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
