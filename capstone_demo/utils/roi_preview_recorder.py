"""
ROI 오버레이를 카메라에 적용해 일정 시간 녹화하고 결과 MP4 파일로 저장하는 스크립트입니다.
사용법 예:
  python utils/roi_preview_recorder.py --roi-json referenceline/roi_data_0.json --duration 6

이 스크립트는 `utils/roi_check.py`의 카메라 해상도/디바이스 탐색 유틸을 재사용합니다.
"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np

# reuse helper from roi_check
from utils import roi_check


def scale_rois_for_frame(rois, frame_w, frame_h):
    max_x = 0
    max_y = 0
    for r in rois:
        for (x, y) in r:
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
    src_w = max_x + 1 if max_x > 0 else frame_w
    src_h = max_y + 1 if max_y > 0 else frame_h
    scale_x = frame_w / src_w
    scale_y = frame_h / src_h
    scaled = []
    for r in rois:
        pts = []
        for (x, y) in r:
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            pts.append((sx, sy))
        scaled.append(np.array(pts, np.int32))
    return scaled


def draw_overlay(frame, roi_payload):
    output = frame.copy()
    overlay = output.copy()

    label_colors = {
        0: (0, 165, 255),
        1: (0, 255, 0),
    }

    for roi_item in roi_payload:
        if not isinstance(roi_item, list) or len(roi_item) != 2:
            continue
        points_data, label = roi_item
        if not isinstance(points_data, list) or len(points_data) < 3:
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in points_data], dtype=np.int32)
        color = label_colors.get(label, (255, 255, 0))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(output, [pts], True, color, 1)

    return cv2.addWeighted(overlay, 0.15, output, 0.85, 0)


def main():
    parser = argparse.ArgumentParser(description="ROI 미리보기 비디오 생성")
    parser.add_argument("--camera", default=None, help="카메라 소스 (인덱스, 경로, 또는 이름)")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON 파일 경로")
    parser.add_argument("--duration", type=float, default=6.0, help="녹화 시간(초)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "referenceline" / "roi_preview.mp4", help="출력 파일 경로")
    parser.add_argument("--width", type=int, default=None, help="강제 캡처 너비")
    parser.add_argument("--height", type=int, default=None, help="강제 캡처 높이")
    args = parser.parse_args()

    # roi json 선택
    if args.roi_json is None:
        # 기본 referenceline 내부의 첫번째 파일 사용
        candidates = sorted((Path(__file__).resolve().parent.parent / "referenceline").glob("roi_data_*.json"))
        if not candidates:
            print("referenceline 폴더에 ROI 파일이 없습니다.")
            return
        roi_json = candidates[0]
    else:
        roi_json = args.roi_json

    try:
        roi_payload = roi_check.load_roi_json(str(roi_json))
    except Exception as e:
        print(f"ROI JSON 로드 실패: {e}")
        return

    cap, cam_path = roi_check.resolve_camera(args.camera)
    if cap is None:
        print("카메라를 열 수 없습니다.")
        return

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 프레임 크기 확인
    ret, frame = cap.read()
    if not ret or frame is None:
        print("카메라에서 프레임을 읽지 못했습니다.")
        cap.release()
        return

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"녹화 시작: {cam_path} -> {out_path} ({w}x{h} @{fps}fps) for {args.duration}s")
    start = time.time()
    frames_written = 0

    try:
        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # ROI를 현재 프레임 크기로 스케일하여 적용
            # roi_payload 형식: [ [ [x,y], ... ], label ]
            # 먼저 단순히 스케일을 위해 원점 기반 포인트 리스트 구성
            rois_points = []
            for item in roi_payload:
                pts, label = item
                rois_points.append(pts)

            scaled_polys = scale_rois_for_frame(rois_points, w, h)

            # scaled_polys는 np.array polygons; 재구성해서 draw_overlay와 동일 포맷으로 사용
            # draw_overlay expects original roi_payload; we'll convert scaled polygons to that format
            scaled_payload = []
            for idx, poly in enumerate(scaled_polys):
                # poly -> list of [x,y]
                pts_list = [[int(p[0]), int(p[1])] for p in poly.tolist()]
                # find label from original payload if possible (by index)
                label = roi_payload[idx][1] if idx < len(roi_payload) and isinstance(roi_payload[idx], list) and len(roi_payload[idx]) == 2 else 0
                scaled_payload.append([pts_list, label])

            display = draw_overlay(frame, scaled_payload)
            writer.write(display)
            frames_written += 1

    finally:
        cap.release()
        writer.release()

    print(f"완료: {frames_written} 프레임을 {out_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
