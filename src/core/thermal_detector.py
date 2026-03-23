"""Thermal anomaly detection."""
import cv2
import numpy as np
from typing import List, Dict, Tuple


class ThermalDetector:
    """Detects thermal anomalies (hot spots) in thermal frames."""

    def __init__(self, config: dict):
        self.threshold = config.get('temperature_threshold', 30.0)
        self.critical_threshold = config.get('critical_threshold', 50.0)
        self.min_area = config.get('min_detection_area', 10)
        self.kernel_size = (5, 5)

    def detect(self, temp_frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect hot spots in temperature frame.

        Returns:
            detections: List of detection dictionaries
            hot_mask: Binary mask of hot regions
        """
        # Create binary mask for pixels above threshold
        hot_mask = (temp_frame > self.threshold).astype(np.uint8) * 255

        # Apply morphological operations to clean up noise
        hot_mask = self._apply_morphological_filters(hot_mask)

        # Find contours of hot regions
        contours, _ = cv2.findContours(
            hot_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract bounding boxes, then merge overlapping ones
        raw_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            raw_bboxes.append(cv2.boundingRect(cnt))

        # Merge overlapping / nearby bounding boxes
        merged_bboxes = self._merge_bboxes(raw_bboxes, margin=15)

        # Build detections from merged bboxes
        detections = []
        for (x, y, w, h) in merged_bboxes:
            roi_temp = temp_frame[y:y+h, x:x+w]
            max_temp = float(np.max(roi_temp))
            mean_temp = float(np.mean(roi_temp))
            min_temp = float(np.min(roi_temp))
            area = int(w * h)

            severity = 'critical' if max_temp > self.critical_threshold else 'warning'

            detection = {
                'bbox': (x, y, w, h),
                'max_temperature': max_temp,
                'mean_temperature': mean_temp,
                'min_temperature': min_temp,
                'area': area,
                'severity': severity,
                'centroid': (x + w // 2, y + h // 2)
            }
            detections.append(detection)

        detections.sort(key=lambda d: d['max_temperature'], reverse=True)
        return detections, hot_mask

    def _merge_bboxes(self, bboxes, margin=15):
        """Merge bounding boxes that overlap or are within margin pixels."""
        if not bboxes:
            return []

        # Convert to x1,y1,x2,y2 with margin expansion
        boxes = []
        for (x, y, w, h) in bboxes:
            boxes.append([x - margin, y - margin, x + w + margin, y + h + margin])

        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = boxes[j]
                    # Check overlap
                    if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used[j] = True
                        merged = True
                new_boxes.append([ax1, ay1, ax2, ay2])
                used[i] = True
            boxes = new_boxes

        # Convert back to (x, y, w, h), removing the margin
        result = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, x1 + margin)
            y1 = max(0, y1 + margin)
            x2 = x2 - margin
            y2 = y2 - margin
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2 - x1, y2 - y1))
        return result

    def _apply_morphological_filters(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up detection mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)

        # Closing: connect nearby regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Opening: remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def update_threshold(self, new_threshold: float):
        """Update detection threshold dynamically."""
        self.threshold = new_threshold
