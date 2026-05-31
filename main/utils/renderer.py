import cv2
import numpy as np

from utils.app_config import DOWNBOUND_LABEL, UPBOUND_LABEL


def draw_roi_overlay(frame, rois):
    if not rois:
        return frame

    output = frame.copy()
    overlay = output.copy()
    label_colors = {
        UPBOUND_LABEL: (0, 255, 0),
        DOWNBOUND_LABEL: (0, 165, 255),
    }

    for points, label in rois:
        color = label_colors.get(label, (255, 255, 0))
        np_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [np_points], color)
        cv2.polylines(output, [np_points], True, color, 2)

    return cv2.addWeighted(overlay, 0.18, output, 0.82, 0)


def draw_tracking_results(frame, detections, rois=None):
    annotated = frame.copy()
    if rois:
        annotated = draw_roi_overlay(annotated, rois)

    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        direction_label = item.get("direction_label")
        if direction_label == UPBOUND_LABEL:
            color = (0, 255, 0)
        elif direction_label == DOWNBOUND_LABEL:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    return annotated
