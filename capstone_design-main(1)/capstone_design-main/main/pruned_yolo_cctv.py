import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

# ultralytics가 홈 디렉터리에 설정 파일을 만들지 못하는 환경 대응
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent


def resolve_local_path(*parts):
    """`main/` 내부 우선, 없으면 프로젝트 루트에서 파일을 찾습니다."""
    local_path = MODULE_DIR.joinpath(*parts)
    if local_path.exists():
        return local_path

    root_path = PROJECT_ROOT.joinpath(*parts)
    if root_path.exists():
        return root_path

    return local_path


# 학습에 사용한 워크스페이스 내 Ultralytics 구현을 우선 사용합니다.
VENDORED_ULTRALYTICS_ROOT = resolve_local_path("yolov12", "yolov12")
if VENDORED_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from ultralytics import YOLO

from roi import build_roi_path, load_roi_data, point_in_rois

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
WEIGHTS_PATH = resolve_local_path("pruned.pt")
YAML_PATH = resolve_local_path("pruned.yaml")
DIRECT_CCTV_URL = ""
DIRECT_CCTV_NAME = ""

TARGET_NAME = "청주"
CCTV_INDEX = 2
VEHICLE_CLASSES = [0]
CONFIDENCE_THRESHOLD = 0.4
ROI_BBOX_OVERLAP_THRESHOLD = 0.3
INFERENCE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2
STREAM_CONNECT_RETRY_COUNT = 2
STREAM_CONNECT_RETRY_DELAY_SECONDS = 0.5
DETECTION_BACKEND_URL = os.environ.get("DETECTION_BACKEND_URL", "").rstrip("/")
DETECTION_SEND_INTERVAL_SECONDS = float(os.environ.get("DETECTION_SEND_INTERVAL", "0.5"))
DETECTION_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DETECTION_REQUEST_TIMEOUT", "3"))
DETECTION_JPEG_QUALITY = int(os.environ.get("DETECTION_JPEG_QUALITY", "80"))

REQUEST_PARAMS = {
    "apiKey": API_KEY,
    "type": "all",
    "cctvType": "1",
    "minX": 127.25,
    "maxX": 127.60,
    "minY": 36.50,
    "maxY": 36.80,
    "getType": "json",
}


def fetch_cctv_dataframe():
    query = "&".join(f"{key}={value}" for key, value in REQUEST_PARAMS.items())
    url = f"{API_URL}?{query}"
    response = urlopen(url)
    payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("response", {}).get("data", [])
    if not data:
        raise RuntimeError("CCTV 목록을 불러오지 못했습니다.")

    cctv_df = pd.json_normalize(data)
    if "cctvname" not in cctv_df.columns:
        raise RuntimeError("CCTV 목록 응답에 cctvname 컬럼이 없습니다.")

    return cctv_df.reset_index(drop=True)


def select_cctv(cctv_df, preferred_index=CCTV_INDEX, target_name=TARGET_NAME):
    print("조회된 CCTV 목록:")
    for index, row in cctv_df.head(LIST_LIMIT).iterrows():
        print(f"[{index}] {row['cctvname']}")

    if preferred_index is not None:
        if preferred_index < 0 or preferred_index >= len(cctv_df):
            raise RuntimeError(f"CCTV_INDEX 범위가 잘못되었습니다: {preferred_index}")

        row = cctv_df.iloc[preferred_index]
        return row["cctvname"], row["cctvurl"], int(preferred_index)

    matched = cctv_df[cctv_df["cctvname"].astype(str).str.contains(target_name, na=False)]
    if matched.empty:
        names = cctv_df["cctvname"].astype(str).dropna().tolist()
        sample_names = ", ".join(names[:10]) if names else "없음"
        raise RuntimeError(
            f"이름으로 CCTV를 찾지 못했습니다: {target_name}\n"
            f"현재 조회된 CCTV 예시: {sample_names}"
        )

    row = matched.iloc[0]
    return row["cctvname"], row["cctvurl"], int(row.name)


def resolve_stream_url(cctv_url):
    try:
        response = requests.get(cctv_url, allow_redirects=True, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"CCTV 스트림 URL 요청 실패: {exc}") from exc

    playlist_url = response.url
    playlist_text = response.text

    for line in playlist_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return urljoin(playlist_url, line)

    return cctv_url


def open_capture(url):
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(url)
    if capture.isOpened():
        return capture

    capture.release()
    return None


def connect_capture(stream_url, cctv_url):
    capture = open_capture(stream_url)
    if capture is not None:
        return capture

    print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
    return open_capture(cctv_url)


def connect_capture_with_retry(cctv_url, retry_count=STREAM_CONNECT_RETRY_COUNT):
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            stream_url = resolve_stream_url(cctv_url)
            capture = connect_capture(stream_url, cctv_url)
            if capture is not None:
                return stream_url, capture
            last_error = RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")
        except RuntimeError as exc:
            last_error = exc

        if attempt < retry_count:
            time.sleep(STREAM_CONNECT_RETRY_DELAY_SECONDS)

    if last_error is None:
        last_error = RuntimeError("알 수 없는 CCTV 연결 오류")
    raise last_error


def read_frame_with_timeout(capture, timeout_seconds):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        ok, frame = capture.read()
        if ok:
            return True, frame

        time.sleep(FRAME_WAIT_INTERVAL_SECONDS)

    return False, None


def load_model():
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"모델 설정 파일을 찾지 못했습니다: {YAML_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"모델 가중치 파일을 찾지 못했습니다: {WEIGHTS_PATH}")

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load
        model = YOLO(str(YAML_PATH), task="detect").load(str(WEIGHTS_PATH))
    finally:
        torch.load = original_torch_load

    if INFERENCE_DEVICE != "cpu":
        model.to(INFERENCE_DEVICE)

    print("YOLO model yaml:", YAML_PATH)
    print("YOLO model weights:", WEIGHTS_PATH)
    print("detection classes:", VEHICLE_CLASSES)
    print("inference device:", INFERENCE_DEVICE)
    return model


def run_inference(model, frame):
    """추적(track) 대신 detection(predict)만 수행해 추가 의존성을 피합니다."""
    return model.predict(
        frame,
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
        device=INFERENCE_DEVICE,
        verbose=False,
    )


def filter_result_by_rois(result, rois):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or not rois:
        return []

    filtered = []
    xyxy_list = boxes.xyxy.int().cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()

    for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
        x1, y1, x2, y2 = xyxy
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        overlap_ratio = get_bbox_roi_overlap_ratio((x1, y1, x2, y2), rois)
        if overlap_ratio >= ROI_BBOX_OVERLAP_THRESHOLD:
            filtered.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "cls_id": int(cls_id),
                    "conf": float(conf),
                    "center": center,
                    "roi_overlap_ratio": overlap_ratio,
                }
            )

    return filtered


def get_bbox_roi_overlap_ratio(xyxy, rois):
    """bbox 면적 중 ROI 내부에 포함되는 비율을 계산합니다."""
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0 or not rois:
        return 0.0

    bbox_area = width * height
    if bbox_area <= 0:
        return 0.0

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    for points, _label in rois:
        shifted_points = np.array(
            [[int(px - x1), int(py - y1)] for px, py in points],
            dtype=np.int32,
        )
        cv2.fillPoly(roi_mask, [shifted_points], 1)

    overlap_pixels = int(np.count_nonzero(roi_mask))
    return min(1.0, overlap_pixels / float(bbox_area))


def draw_roi_overlay(frame, rois):
    if not rois:
        return frame

    output = frame.copy()
    overlay = output.copy()
    label_colors = {
        0: (0, 255, 0),
        1: (0, 165, 255),
    }

    for points, label in rois:
        color = label_colors.get(label, (255, 255, 0))
        np_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [np_points], color)
        cv2.polylines(output, [np_points], True, color, 2)

    return cv2.addWeighted(overlay, 0.18, output, 0.82, 0)


def collect_detections(result, rois=None):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    roi_filtered_boxes = None
    if rois:
        roi_filtered_boxes = filter_result_by_rois(result, rois)

    if roi_filtered_boxes is not None:
        iterable_boxes = roi_filtered_boxes
    else:
        xyxy_list = boxes.xyxy.int().cpu().tolist()
        cls_list = boxes.cls.int().cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()
        iterable_boxes = [
            {
                "xyxy": tuple(xyxy),
                "cls_id": int(cls_id),
                "conf": float(conf),
            }
            for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list)
        ]

    names = result.names or {}
    detections = []
    for item in iterable_boxes:
        cls_id = int(item["cls_id"])
        detection = {
            "class_name": str(names.get(cls_id, cls_id)),
            "class_id": cls_id,
            "confidence": round(float(item["conf"]), 4),
            "bbox": [int(value) for value in item["xyxy"]],
        }
        if "roi_overlap_ratio" in item:
            detection["roi_overlap_ratio"] = round(float(item["roi_overlap_ratio"]), 4)
        detections.append(detection)

    return detections


def draw_tracking_results(frame, detections, rois=None):
    annotated = frame.copy()
    if rois:
        annotated = draw_roi_overlay(annotated, rois)

    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    return annotated


def post_detection_to_backend(
    backend_url,
    frame,
    detections,
    traffic_count,
    selected_name,
    selected_index,
    stream_status,
    roi_enabled,
    roi_path,
):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), DETECTION_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")

    best_detection = max(
        detections,
        key=lambda item: float(item.get("confidence", 0) or 0),
        default={"class_name": "none", "confidence": 0.0, "bbox": [0, 0, 0, 0]},
    )
    data = {
        "class_name": best_detection["class_name"],
        "confidence": str(best_detection["confidence"]),
        "bbox": json.dumps(best_detection["bbox"]),
        "detections": json.dumps(detections, ensure_ascii=False),
        "traffic_count": str(traffic_count),
        "selected_name": selected_name,
        "selected_index": str(selected_index),
        "stream_status": stream_status,
        "roi_enabled": str(bool(roi_enabled)).lower(),
        "roi_path": str(roi_path or ""),
    }
    files = {
        "frame": ("frame.jpg", encoded.tobytes(), "image/jpeg"),
    }
    response = requests.post(
        f"{backend_url}/api/detection",
        data=data,
        files=files,
        timeout=DETECTION_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


class PrunedCCTVPipeline:
    """CCTV 수신, Pruned YOLO 추론, ROI 반영까지 담당하는 메인 파이프라인."""

    def __init__(
        self,
        cctv_index=None,
        cctv_url="",
        cctv_name="",
        roi_path=None,
        use_roi=False,
    ):
        self.model = load_model()
        self.cctv_df = fetch_cctv_dataframe()
        self.cctv_items = self.cctv_df["cctvname"].astype(str).fillna("").tolist()

        self.cctv_index = CCTV_INDEX if cctv_index is None else cctv_index
        self.direct_cctv_url = cctv_url or DIRECT_CCTV_URL
        self.direct_cctv_name = cctv_name or DIRECT_CCTV_NAME
        self.use_roi = bool(use_roi)
        self._custom_roi_path = roi_path
        self.roi_path = roi_path
        self.rois = []

        self.capture = None
        self.current_stream_url = None
        self.current_cctv_name = "-"
        self.current_cctv_url = ""
        self.current_index = 0
        self.remote_backend_url = DETECTION_BACKEND_URL
        self.remote_send_interval = DETECTION_SEND_INTERVAL_SECONDS
        self._last_remote_send_at = 0.0
        self._remote_send_in_flight = False
        self._remote_send_lock = threading.Lock()

    def get_cctv_items(self):
        return [
            {
                "index": index,
                "name": name,
                "selected": index == self.current_index,
            }
            for index, name in enumerate(self.cctv_items)
        ]

    def open(self):
        if self.direct_cctv_url:
            self.current_cctv_name = self.direct_cctv_name or "직접 지정 CCTV"
            self.current_cctv_url = self.direct_cctv_url
            self.current_index = 0
        else:
            cctv_name, cctv_url, selected_index = select_cctv(self.cctv_df, self.cctv_index)
            self.current_cctv_name = str(cctv_name)
            self.current_cctv_url = str(cctv_url)
            self.current_index = int(selected_index)

        self._update_roi_path()
        self._open_capture()
        return self.current_cctv_name

    def select_cctv(self, index):
        if index < 0 or index >= len(self.cctv_df):
            raise IndexError("CCTV 인덱스 범위를 벗어났습니다.")

        previous_index = self.current_index
        previous_name = self.current_cctv_name
        previous_url = self.current_cctv_url
        previous_roi_path = self.roi_path
        previous_rois = list(self.rois)
        previous_stream_url = self.current_stream_url
        previous_capture = self.capture

        row = self.cctv_df.iloc[index]
        self.current_index = int(index)
        self.current_cctv_name = str(row["cctvname"])
        self.current_cctv_url = str(row["cctvurl"])
        self._update_roi_path()
        self.capture = None

        try:
            self._open_capture()
        except Exception:
            self.current_index = previous_index
            self.current_cctv_name = previous_name
            self.current_cctv_url = previous_url
            self.roi_path = previous_roi_path
            self.rois = previous_rois
            self.current_stream_url = previous_stream_url
            self.capture = previous_capture
            raise

        if previous_capture is not None:
            previous_capture.release()

    def reload_rois(self):
        if self.roi_path and self.use_roi:
            self.rois = load_roi_data(self.roi_path)
        else:
            self.rois = []

    def _update_roi_path(self):
        if self._custom_roi_path:
            self.roi_path = self._custom_roi_path
        else:
            self.roi_path = build_roi_path(self.current_index)
        self.reload_rois()

    def _open_capture(self):
        if not self.current_cctv_url:
            raise RuntimeError("현재 CCTV URL이 비어 있습니다.")

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.current_stream_url, self.capture = connect_capture_with_retry(self.current_cctv_url)

    def _send_detection_if_needed(self, frame, detections, traffic_count):
        if not self.remote_backend_url:
            return

        now = time.time()
        if now - self._last_remote_send_at < self.remote_send_interval:
            return

        with self._remote_send_lock:
            if self._remote_send_in_flight:
                return
            self._remote_send_in_flight = True
            self._last_remote_send_at = now

        payload_frame = frame.copy()
        payload_detections = [dict(item) for item in detections]
        payload = {
            "backend_url": self.remote_backend_url,
            "frame": payload_frame,
            "detections": payload_detections,
            "traffic_count": int(traffic_count),
            "selected_name": self.current_cctv_name,
            "selected_index": int(self.current_index),
            "stream_status": "연결됨",
            "roi_enabled": bool(self.rois if self.use_roi else []),
            "roi_path": self.roi_path,
        }

        thread = threading.Thread(
            target=self._send_detection_worker,
            kwargs=payload,
            daemon=True,
        )
        thread.start()

    def _send_detection_worker(self, **payload):
        try:
            post_detection_to_backend(**payload)
        except requests.RequestException as exc:
            print(f"remote detection send failed: {exc}")
        except Exception as exc:
            print(f"remote detection send failed: {exc}")
        finally:
            with self._remote_send_lock:
                self._remote_send_in_flight = False

    def process_next_frame(self):
        if self.capture is None or not self.capture.isOpened():
            self._open_capture()

        ok, frame = read_frame_with_timeout(self.capture, FRAME_WAIT_TIMEOUT_SECONDS)
        if not ok:
            if self.capture is not None:
                self.capture.release()
            time.sleep(1)
            self._open_capture()
            ok, frame = read_frame_with_timeout(self.capture, FRAME_WAIT_TIMEOUT_SECONDS)
            if not ok:
                raise RuntimeError("프레임 수신 실패")

        results = run_inference(self.model, frame)
        rois = self.rois if self.use_roi else []
        detections = collect_detections(results[0], rois=rois)
        annotated_frame = draw_tracking_results(frame, detections, rois=rois)
        traffic_count = len(detections)
        self._send_detection_if_needed(annotated_frame, detections, traffic_count)

        return {
            "frame": annotated_frame,
            "traffic_count": traffic_count,
            "detections": detections,
            "selected_name": self.current_cctv_name,
            "selected_index": self.current_index,
            "stream_url": self.current_stream_url,
            "stream_status": "연결됨",
            "yolo_enabled": True,
            "roi_enabled": bool(rois),
            "roi_path": self.roi_path,
        }

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def main():
    pipeline = PrunedCCTVPipeline()
    cctv_name = pipeline.open()
    print("선택 CCTV:", cctv_name)
    print("cctvurl:", pipeline.current_cctv_url)
    print("stream url:", pipeline.current_stream_url)

    try:
        while True:
            output = pipeline.process_next_frame()
            cv2.imshow("ITS CCTV Live", output["frame"])

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
