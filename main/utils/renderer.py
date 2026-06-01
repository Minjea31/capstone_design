import cv2
import numpy as np

from utils.app_config import DOWNBOUND_LABEL, UPBOUND_LABEL


def _direction_color(label):
    if label == UPBOUND_LABEL:
        return (0, 255, 0)
    if label == DOWNBOUND_LABEL:
        return (0, 165, 255)
    return (255, 255, 0)


def _normalize_roi(roi):
    if isinstance(roi, dict):
        points = roi.get("points", [])
        label = roi.get("label", -1)
    else:
        points, label = roi

    polygon = np.array(points, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return None
    return polygon, int(label)


def _draw_rois(frame, rois):
    if not rois:
        return frame

    overlay = frame.copy()
    for roi in rois:
        normalized = _normalize_roi(roi)
        if normalized is None:
            continue
        polygon, label = normalized
        color = _direction_color(label)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(overlay, [polygon], True, color, 1)

    return cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)


def draw_tracking_results(frame, detections, rois=None):
    annotated = frame.copy()
    annotated = _draw_rois(annotated, rois)

    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        direction_label = item.get("direction_label")
        color = _direction_color(direction_label)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    return annotated
