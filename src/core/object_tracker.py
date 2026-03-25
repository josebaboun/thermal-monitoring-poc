"""Object tracking for thermal detections across frames."""
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict
from scipy.spatial import distance as dist


def _bbox_iou(box_a, box_b):
    """Compute IoU between two bboxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to x1, y1, x2, y2
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class ThermalObjectTracker:
    """Tracks thermal objects across frames to avoid duplicate alerts."""

    def __init__(self, max_disappeared: int = 75, max_distance: float = 150.0,
                 min_confirm_frames: int = 5, critical_threshold: float = 60.0):
        self.next_object_id = 0
        self.objects = OrderedDict()       # {object_id: centroid}
        self.bboxes = OrderedDict()        # {object_id: bbox}
        self.disappeared = OrderedDict()   # {object_id: frame_count}
        self.object_info = OrderedDict()   # {object_id: {...}}
        self.confirmed = set()             # object_ids confirmed as real

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_confirm_frames = min_confirm_frames
        self.critical_threshold = critical_threshold

    def register(self, centroid: Tuple[int, int], detection: Dict, frame: int) -> int:
        """Register a new object."""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = detection['bbox']
        self.disappeared[object_id] = 0
        self.object_info[object_id] = {
            'first_seen_frame': frame,
            'last_seen_frame': frame,
            'max_temperature': detection['max_temperature'],
            'mean_temperature': detection['mean_temperature'],
            '_temp_sum': detection['max_temperature'],
            'severity': detection['severity'],
            'bbox': detection['bbox'],
            'total_detections': 1
        }
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        """Remove object from active tracking (keep info for report)."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def _compute_cost_matrix(self, object_ids, object_centroids, input_centroids, detections):
        """Compute cost matrix combining centroid distance and IoU."""
        n_objects = len(object_ids)
        n_detections = len(detections)
        cost = np.full((n_objects, n_detections), float('inf'))

        for i, obj_id in enumerate(object_ids):
            obj_bbox = self.bboxes[obj_id]
            for j, det in enumerate(detections):
                # Centroid distance (normalized to 0-1 range over max_distance)
                cdist = np.linalg.norm(
                    np.array(object_centroids[i]) - np.array(input_centroids[j])
                )
                if cdist > self.max_distance:
                    continue

                # IoU bonus: higher IoU = lower cost
                iou = _bbox_iou(obj_bbox, det['bbox'])

                # Combined cost: distance penalty - IoU bonus
                # Distance normalized by max_distance, IoU in [0,1]
                cost[i, j] = (cdist / self.max_distance) - iou

        return cost

    def update(self, detections: List[Dict], frame: int) -> Dict[int, Dict]:
        """Update tracker with new detections."""
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.array([det['centroid'] for det in detections])

        # If no existing objects, register all as new
        if len(self.objects) == 0:
            new_objects = {}
            for i, detection in enumerate(detections):
                obj_id = self.register(input_centroids[i], detection, frame)
                new_objects[obj_id] = self.object_info[obj_id]
            return new_objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute cost matrix (distance + IoU)
        cost = self._compute_cost_matrix(
            object_ids, object_centroids, input_centroids, detections
        )

        # Greedy matching: assign lowest cost first
        used_rows = set()
        used_cols = set()
        matched_objects = {}

        # Flatten and sort by cost
        indices = []
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if cost[i, j] < float('inf'):
                    indices.append((cost[i, j], i, j))
        indices.sort()

        for _, row, col in indices:
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.bboxes[object_id] = detections[col]['bbox']
            self.disappeared[object_id] = 0

            info = self.object_info[object_id]
            info['last_seen_frame'] = frame
            info['total_detections'] += 1

            # Track running average temperature for robust severity
            det_temp = detections[col]['max_temperature']
            info['_temp_sum'] += det_temp
            avg_temp = info['_temp_sum'] / info['total_detections']
            info['max_temperature'] = avg_temp
            info['severity'] = detections[col]['severity'] if avg_temp <= info['max_temperature'] else info['severity']
            # Re-evaluate severity based on average temperature
            info['severity'] = 'critical' if avg_temp > self.critical_threshold else 'warning'

            info['bbox'] = detections[col]['bbox']
            matched_objects[object_id] = info

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in set(range(cost.shape[0])) - used_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Register unmatched detections as new (unconfirmed) objects
        for col in set(range(cost.shape[1])) - used_cols:
            self.register(input_centroids[col], detections[col], frame)

        # Check for newly confirmed objects (reached min_confirm_frames)
        newly_confirmed = {}
        for obj_id in list(self.objects.keys()):
            if obj_id not in self.confirmed:
                info = self.object_info[obj_id]
                if info['total_detections'] >= self.min_confirm_frames:
                    self.confirmed.add(obj_id)
                    newly_confirmed[obj_id] = info

        # Return only confirmed objects (newly confirmed + matched confirmed)
        confirmed_matched = {
            k: v for k, v in matched_objects.items() if k in self.confirmed
        }
        return {**newly_confirmed, **confirmed_matched}

    def get_active_objects(self) -> Dict[int, Dict]:
        """Get all currently tracked objects."""
        return {
            obj_id: info
            for obj_id, info in self.object_info.items()
            if obj_id in self.objects
        }

    def get_all_objects(self) -> Dict[int, Dict]:
        """Get all confirmed objects (including disappeared)."""
        return {
            obj_id: info
            for obj_id, info in self.object_info.items()
            if obj_id in self.confirmed
        }

    def get_new_objects_since_last_update(self, last_object_id: int) -> Dict[int, Dict]:
        """Get objects that were registered after a certain ID."""
        return {
            obj_id: info
            for obj_id, info in self.object_info.items()
            if obj_id > last_object_id
        }

    def get_object_count(self) -> Tuple[int, int]:
        """Get (active_confirmed_count, total_confirmed_count)."""
        active = sum(1 for obj_id in self.objects if obj_id in self.confirmed)
        total = len(self.confirmed)
        return active, total
