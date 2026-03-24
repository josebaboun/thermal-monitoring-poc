"""Video rendering with thermal overlays and detection visualization."""
import cv2
import numpy as np
from typing import List, Dict, Optional


class VideoRenderer:
    """Renders thermal video with detection overlays."""

    def __init__(self, config: dict):
        self.show_bbox = config.get('show_bbox', True)
        self.show_temperature = config.get('show_temperature_text', True)
        self.show_timestamp = config.get('show_timestamp', True)
        self.bbox_color = tuple(config.get('bbox_color', [0, 0, 255]))
        self.bbox_thickness = config.get('bbox_thickness', 2)
        self.font_scale = config.get('font_scale', 0.6)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        temp_frame: Optional[np.ndarray],
        detections: List[Dict],
        timestamp: str = "00:00.000"
    ) -> np.ndarray:
        """
        Render frame with all overlays.

        Args:
            frame: Original frame (RGB)
            temp_frame: Temperature frame (for statistics)
            detections: List of detections
            timestamp: Video timestamp

        Returns:
            Rendered frame with overlays
        """
        output_frame = frame.copy()

        # Draw detections
        if self.show_bbox and detections:
            output_frame = self._draw_detections(output_frame, detections)

        # Draw timestamp
        if self.show_timestamp and timestamp:
            output_frame = self._draw_timestamp(output_frame, timestamp)

        # Draw detection count
        if detections:
            output_frame = self._draw_detection_count(output_frame, len(detections))

        return output_frame

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and temperature text for detections."""
        for det in detections:
            x, y, w, h = det['bbox']
            max_temp = det['max_temperature']
            severity = det['severity']

            # Color based on severity
            if severity == 'critical':
                color = (0, 0, 255)  # Red
                label_bg_color = (0, 0, 255)
            else:
                color = (0, 165, 255)  # Orange
                label_bg_color = (0, 165, 255)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                self.bbox_thickness
            )

            # Draw temperature text
            if self.show_temperature:
                temp_text = f"{max_temp:.1f}C"
                if severity == 'critical':
                    temp_text = f"! {temp_text} !"

                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    temp_text,
                    self.font,
                    self.font_scale,
                    2
                )

                # Draw background rectangle for text
                text_x = x
                text_y = y - 10 if y - 10 > text_height else y + h + text_height + 10

                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y + baseline),
                    label_bg_color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    frame,
                    temp_text,
                    (text_x, text_y),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),  # White text
                    2,
                    cv2.LINE_AA
                )

        return frame

    def _draw_timestamp(self, frame: np.ndarray, timestamp: str) -> np.ndarray:
        """Draw timestamp overlay."""
        h, w = frame.shape[:2]

        # Position in top-right corner
        text = f"Time: {timestamp}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            2
        )

        # Background
        padding = 10
        cv2.rectangle(
            frame,
            (w - text_width - padding * 2, padding),
            (w - padding, text_height + padding * 2),
            (0, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            text,
            (w - text_width - padding, text_height + padding),
            self.font,
            self.font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    def _draw_detection_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """Draw detection count overlay."""
        text = f"Detections: {count}"

        # Position in top-left corner
        padding = 10
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            2
        )

        # Background
        cv2.rectangle(
            frame,
            (padding, padding),
            (text_width + padding * 2, text_height + padding * 2),
            (0, 0, 0),
            -1
        )

        # Text
        color = (0, 255, 0) if count == 0 else (0, 165, 255)
        cv2.putText(
            frame,
            text,
            (padding * 2, text_height + padding),
            self.font,
            self.font_scale,
            color,
            2,
            cv2.LINE_AA
        )

        return frame
