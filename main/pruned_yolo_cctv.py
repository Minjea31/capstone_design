import json
import os
import sys
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
CONFIDENCE_THRESHOLD = 0.3
LIST_LIMIT = 20
FRAME_WAIT_TIMEOUT_SECONDS = 10
FRAME_WAIT_INTERVAL_SECONDS = 0.2

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

    return pd.json_normalize(data)


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
    response = requests.get(cctv_url, allow_redirects=True, timeout=10)
    response.raise_for_status()

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

    print("YOLO model yaml:", YAML_PATH)
    print("YOLO model weights:", WEIGHTS_PATH)
    print("detection classes:", VEHICLE_CLASSES)
    return model


def run_inference(model, frame):
    """추적(track) 대신 detection(predict)만 수행해 추가 의존성을 피합니다."""
    return model.predict(
        frame,
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
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
        if point_in_rois(center, rois):
            filtered.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "cls_id": int(cls_id),
                    "conf": float(conf),
                    "center": center,
                }
            )

    return filtered


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


def draw_tracking_results(frame, result, rois=None):
    annotated = frame.copy()
    if rois:
        annotated = draw_roi_overlay(annotated, rois)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, 0

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

    count = 0
    for item in iterable_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        conf = item["conf"]
        color = (0, 255, 0)
        text = f"{conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        count += 1

    return annotated, count


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

        row = self.cctv_df.iloc[index]
        self.current_index = int(index)
        self.current_cctv_name = str(row["cctvname"])
        self.current_cctv_url = str(row["cctvurl"])
        self._update_roi_path()
        self._open_capture()

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
        if self.capture is not None:
            self.capture.release()

        self.current_stream_url = resolve_stream_url(self.current_cctv_url)
        self.capture = connect_capture(self.current_stream_url, self.current_cctv_url)

        if self.capture is None:
            raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

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
        annotated_frame, traffic_count = draw_tracking_results(frame, results[0], rois=rois)

        return {
            "frame": annotated_frame,
            "traffic_count": traffic_count,
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
