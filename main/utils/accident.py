import time
from math import hypot, sqrt

from utils.app_config import (
    ACCIDENT_BBOX_IOU_THRESHOLD,
    ACCIDENT_BBOX_MIN_RATIO_THRESHOLD,
    ACCIDENT_COLLISION_MOTION_MIN_SECONDS,
    ACCIDENT_COLLISION_MOTION_THRESHOLD,
    ACCIDENT_CONGESTION_SUPPRESS_MIN_VEHICLES,
    ACCIDENT_DETECTION_ENABLED,
    ACCIDENT_DUPLICATE_CENTER_DISTANCE_RATIO,
    ACCIDENT_DUPLICATE_IOU_THRESHOLD,
    ACCIDENT_DUPLICATE_MIN_RATIO_THRESHOLD,
    ACCIDENT_LOG_REPEAT_SECONDS,
    ACCIDENT_ON_THRESHOLD,
    ACCIDENT_RED_HOLD_SECONDS,
    ACCIDENT_SCORE_DEC,
    ACCIDENT_SCORE_INC,
    ACCIDENT_SAME_DIRECTION_IOU_THRESHOLD,
    ACCIDENT_SAME_DIRECTION_MIN_RATIO_THRESHOLD,
    ACCIDENT_STOPPED_OVERLAP_MIN_STATIONARY_FRAMES,
    ACCIDENT_STOPPED_OVERLAP_MOVEMENT_THRESHOLD,
)


def _bbox_overlap_metrics(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = [float(value) for value in bbox_a]
    bx1, by1, bx2, by2 = [float(value) for value in bbox_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    min_area = min(area_a, area_b)

    return {
        "intersection": inter_area,
        "iou": inter_area / union_area if union_area > 0 else 0.0,
        "min_ratio": inter_area / min_area if min_area > 0 else 0.0,
    }


def _bbox_center(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _passes_overlap_threshold(overlap, same_direction):
    if same_direction:
        return (
            overlap["iou"] >= ACCIDENT_SAME_DIRECTION_IOU_THRESHOLD
            or overlap["min_ratio"] >= ACCIDENT_SAME_DIRECTION_MIN_RATIO_THRESHOLD
        )
    return (
        overlap["iou"] >= ACCIDENT_BBOX_IOU_THRESHOLD
        or overlap["min_ratio"] >= ACCIDENT_BBOX_MIN_RATIO_THRESHOLD
    )


def _direction_pair(detection_a, detection_b):
    direction_a = detection_a.get("direction_label")
    direction_b = detection_b.get("direction_label")
    if direction_a is None or direction_b is None:
        return direction_a, direction_b, False, False

    same_direction = int(direction_a) == int(direction_b)
    different_direction = not same_direction
    return int(direction_a), int(direction_b), same_direction, different_direction


def _detection_center(detection):
    center = detection.get("center")
    if center is not None:
        return (float(center[0]), float(center[1]))

    bbox = detection.get("bbox")
    if bbox:
        return _bbox_center(bbox)

    return None


def _track_key(detection, fallback_index):
    track_id = detection.get("track_id")
    if track_id is not None:
        return f"track:{int(track_id)}", int(track_id)

    bbox = detection.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return f"bbox:{center_x // 16}:{center_y // 16}:{fallback_index}", None


def _looks_like_duplicate_detection(overlap, detection_a, detection_b):
    if overlap["iou"] < ACCIDENT_DUPLICATE_IOU_THRESHOLD:
        return False
    if overlap["min_ratio"] < ACCIDENT_DUPLICATE_MIN_RATIO_THRESHOLD:
        return False

    bbox_a = detection_a.get("bbox")
    bbox_b = detection_b.get("bbox")
    if not bbox_a or not bbox_b:
        return False

    center_a = _detection_center(detection_a)
    center_b = _detection_center(detection_b)
    if center_a is None or center_b is None:
        return False

    min_area = min(_bbox_area(bbox_a), _bbox_area(bbox_b))
    if min_area <= 0:
        return False

    center_distance = hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    return center_distance <= sqrt(min_area) * ACCIDENT_DUPLICATE_CENTER_DISTANCE_RATIO


class AccidentLogDetector:
    """Log likely accidents from YOLO detection pairs without changing payloads."""

    def __init__(self):
        self._scores = {}
        self._active = {}
        self._active_until = {}
        self._last_log_at = {}
        self._last_candidates = {}
        self._pair_motion_states = {}
        self._track_states = {}

    def reset(self):
        self._scores.clear()
        self._active.clear()
        self._active_until.clear()
        self._last_log_at.clear()
        self._last_candidates.clear()
        self._pair_motion_states.clear()
        self._track_states.clear()

    def update(self, detections, frame_id, cctv_name="-", timestamp=""):
        if not ACCIDENT_DETECTION_ENABLED:
            return []

        now = time.time()
        self._update_track_states(detections, frame_id)
        self._update_pair_motion_states(detections, now)
        candidates = self._collect_candidates(detections)
        active_events = []

        for pair_key in set(self._scores) | set(candidates) | set(self._active_until):
            candidate = candidates.get(pair_key)
            if candidate is not None:
                self._last_candidates[pair_key] = candidate
                score = min(ACCIDENT_ON_THRESHOLD, self._scores.get(pair_key, 0.0) + ACCIDENT_SCORE_INC)
            else:
                score = max(0.0, self._scores.get(pair_key, 0.0) - ACCIDENT_SCORE_DEC)

            self._scores[pair_key] = score
            if candidate is None:
                held_event = self._build_held_event(pair_key, frame_id, cctv_name, timestamp, score, now)
                if held_event is not None:
                    active_events.append(held_event)
                    continue
                if score < ACCIDENT_ON_THRESHOLD:
                    self._active[pair_key] = False
                if score <= 0.0:
                    self._scores.pop(pair_key, None)
                    self._active.pop(pair_key, None)
                    self._active_until.pop(pair_key, None)
                    self._last_log_at.pop(pair_key, None)
                    self._last_candidates.pop(pair_key, None)
                continue

            active_before = self._active.get(pair_key, False)
            active_now = score >= ACCIDENT_ON_THRESHOLD
            self._active[pair_key] = active_now

            if not active_now:
                continue

            self._active_until[pair_key] = max(
                self._active_until.get(pair_key, 0.0),
                now + ACCIDENT_RED_HOLD_SECONDS,
            )

            event = {
                **candidate,
                "frame_id": int(frame_id),
                "score": round(float(score), 2),
                "cctv_name": str(cctv_name),
                "timestamp": str(timestamp),
            }
            active_events.append(event)

            last_log_at = self._last_log_at.get(pair_key, 0.0)
            should_log = (not active_before) or (now - last_log_at >= ACCIDENT_LOG_REPEAT_SECONDS)
            if not should_log:
                continue

            self._last_log_at[pair_key] = now
            self._log_event(event)

        return active_events

    def _build_held_event(self, pair_key, frame_id, cctv_name, timestamp, score, now):
        if now > self._active_until.get(pair_key, 0.0):
            return None

        candidate = self._last_candidates.get(pair_key)
        if candidate is None:
            return None

        return {
            **candidate,
            "frame_id": int(frame_id),
            "score": round(float(score), 2),
            "cctv_name": str(cctv_name),
            "timestamp": str(timestamp),
            "held": True,
        }

    def _update_track_states(self, detections, frame_id):
        current_keys = set()

        for index, detection in enumerate(detections):
            key, _track_id = _track_key(detection, index)
            center = _detection_center(detection)
            if center is None:
                continue

            current_keys.add(key)
            previous = self._track_states.get(key)
            movement = None
            stationary_frames = 0
            if previous and previous.get("center") is not None:
                previous_center = previous["center"]
                movement = hypot(center[0] - previous_center[0], center[1] - previous_center[1])
                if movement <= ACCIDENT_STOPPED_OVERLAP_MOVEMENT_THRESHOLD:
                    stationary_frames = int(previous.get("stationary_frames", 0)) + 1

            self._track_states[key] = {
                "center": center,
                "movement": movement,
                "stationary_frames": stationary_frames,
                "last_frame_id": int(frame_id),
            }

        stale_keys = [
            key
            for key, state in self._track_states.items()
            if key not in current_keys and int(frame_id) - int(state.get("last_frame_id", frame_id)) > 30
        ]
        for key in stale_keys:
            self._track_states.pop(key, None)

    def _update_pair_motion_states(self, detections, now):
        seen_pair_keys = set()

        for index_a, detection_a in enumerate(detections):
            bbox_a = detection_a.get("bbox")
            if not bbox_a:
                continue

            key_a, _track_id_a = _track_key(detection_a, index_a)
            for index_b in range(index_a + 1, len(detections)):
                detection_b = detections[index_b]
                bbox_b = detection_b.get("bbox")
                if not bbox_b:
                    continue

                key_b, _track_id_b = _track_key(detection_b, index_b)
                pair_key = tuple(sorted((key_a, key_b)))
                overlap = _bbox_overlap_metrics(bbox_a, bbox_b)
                direction_a, direction_b, same_direction, _different_direction = _direction_pair(detection_a, detection_b)
                if not _passes_overlap_threshold(overlap, same_direction):
                    continue
                if same_direction and _looks_like_duplicate_detection(overlap, detection_a, detection_b):
                    continue

                seen_pair_keys.add(pair_key)
                state = self._pair_motion_states.get(pair_key, {})
                previous_seen_at = state.get("last_seen_at")
                elapsed = max(0.0, min(1.0, now - previous_seen_at)) if previous_seen_at is not None else 0.0
                movement_a = (self._track_states.get(key_a) or {}).get("movement") or 0.0
                movement_b = (self._track_states.get(key_b) or {}).get("movement") or 0.0
                moving = max(movement_a, movement_b) > ACCIDENT_COLLISION_MOTION_THRESHOLD
                moving_overlap_seconds = float(state.get("moving_overlap_seconds", 0.0))
                if moving:
                    moving_overlap_seconds += elapsed

                self._pair_motion_states[pair_key] = {
                    "directions": (direction_a, direction_b),
                    "last_seen_at": now,
                    "moving_overlap_seconds": moving_overlap_seconds,
                }

        stale_pair_keys = [
            pair_key
            for pair_key, state in self._pair_motion_states.items()
            if pair_key not in seen_pair_keys and now - float(state.get("last_seen_at", now)) > 2.0
        ]
        for pair_key in stale_pair_keys:
            self._pair_motion_states.pop(pair_key, None)

    def _collision_motion_ready(self, pair_key):
        state = self._pair_motion_states.get(pair_key) or {}
        return float(state.get("moving_overlap_seconds", 0.0)) >= ACCIDENT_COLLISION_MOTION_MIN_SECONDS

    def _stopped_overlap_ready(self, key_a, key_b):
        state_a = self._track_states.get(key_a) or {}
        state_b = self._track_states.get(key_b) or {}
        stationary_a = int(state_a.get("stationary_frames", 0))
        stationary_b = int(state_b.get("stationary_frames", 0))
        return (
            stationary_a >= ACCIDENT_STOPPED_OVERLAP_MIN_STATIONARY_FRAMES
            and stationary_b >= ACCIDENT_STOPPED_OVERLAP_MIN_STATIONARY_FRAMES
        )

    def _stopped_vehicle_counts_by_direction(self, detections):
        counts = {}
        for index, detection in enumerate(detections):
            direction = detection.get("direction_label")
            if direction is None:
                continue

            key, _track_id = _track_key(detection, index)
            state = self._track_states.get(key) or {}
            if int(state.get("stationary_frames", 0)) < ACCIDENT_STOPPED_OVERLAP_MIN_STATIONARY_FRAMES:
                continue

            direction = int(direction)
            counts[direction] = counts.get(direction, 0) + 1

        return counts

    def _inside_same_direction_congestion(self, direction, stopped_counts):
        if direction is None:
            return False
        return int(stopped_counts.get(int(direction), 0)) >= ACCIDENT_CONGESTION_SUPPRESS_MIN_VEHICLES

    def _collect_candidates(self, detections):
        candidates = {}
        if len(detections) < 2:
            return candidates

        stopped_counts = self._stopped_vehicle_counts_by_direction(detections)

        for index_a, detection_a in enumerate(detections):
            bbox_a = detection_a.get("bbox")
            if not bbox_a:
                continue

            key_a, track_id_a = _track_key(detection_a, index_a)
            for index_b in range(index_a + 1, len(detections)):
                detection_b = detections[index_b]
                bbox_b = detection_b.get("bbox")
                if not bbox_b:
                    continue

                key_b, track_id_b = _track_key(detection_b, index_b)
                pair_key = tuple(sorted((key_a, key_b)))
                overlap = _bbox_overlap_metrics(bbox_a, bbox_b)
                direction_a, direction_b, same_direction, different_direction = _direction_pair(detection_a, detection_b)
                if not _passes_overlap_threshold(overlap, same_direction):
                    continue

                if same_direction:
                    if _looks_like_duplicate_detection(overlap, detection_a, detection_b):
                        continue
                    if not self._stopped_overlap_ready(key_a, key_b):
                        continue
                    if not self._collision_motion_ready(pair_key):
                        continue
                    if self._inside_same_direction_congestion(direction_a, stopped_counts):
                        continue
                    reason = "same_direction_collision_stopped_overlap"
                elif different_direction:
                    if not self._stopped_overlap_ready(key_a, key_b):
                        continue
                    if not self._collision_motion_ready(pair_key):
                        continue
                    reason = "different_direction_collision_stopped_overlap"
                else:
                    if not self._stopped_overlap_ready(key_a, key_b):
                        continue
                    if not self._collision_motion_ready(pair_key):
                        continue
                    reason = "unknown_direction_collision_stopped_overlap"

                candidates[pair_key] = {
                    "pair_key": pair_key,
                    "detection_indexes": (index_a, index_b),
                    "track_ids": (track_id_a, track_id_b),
                    "directions": (direction_a, direction_b),
                    "reason": reason,
                    "iou": round(float(overlap["iou"]), 4),
                    "min_overlap": round(float(overlap["min_ratio"]), 4),
                    "stationary_frames": (
                        int((self._track_states.get(key_a) or {}).get("stationary_frames", 0)),
                        int((self._track_states.get(key_b) or {}).get("stationary_frames", 0)),
                    ),
                    "movements": (
                        (self._track_states.get(key_a) or {}).get("movement"),
                        (self._track_states.get(key_b) or {}).get("movement"),
                    ),
                    "moving_overlap_seconds": round(
                        float((self._pair_motion_states.get(pair_key) or {}).get("moving_overlap_seconds", 0.0)),
                        3,
                    ),
                    "bbox_a": [int(value) for value in bbox_a],
                    "bbox_b": [int(value) for value in bbox_b],
                }

        return candidates

    def _log_event(self, event):
        print(
            "[accident_detect]",
            f"frame={event['frame_id']}",
            f"cctv={event['cctv_name']}",
            f"tracks={event['track_ids']}",
            f"directions={event['directions']}",
            f"reason={event['reason']}",
            f"score={event['score']:.1f}",
            f"iou={event['iou']:.2f}",
            f"min_overlap={event['min_overlap']:.2f}",
            f"stationary={event.get('stationary_frames', '-')}",
            f"moving_overlap={event.get('moving_overlap_seconds', 0.0):.2f}s",
            f"time={event['timestamp']}",
            flush=True,
        )
