import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "ROI Check - USB Camera"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "roi_check_output"


def camera_name(device_path):
    """카메라 장치의 이름을 가져옵니다."""
    device_name = Path(os.path.realpath(device_path)).name
    name_path = Path("/sys/class/video4linux") / device_name / "name"

    try:
        return name_path.read_text().strip()
    except OSError:
        return device_name


def camera_candidates():
    """사용 가능한 카메라 장치 목록을 우선순위 순으로 정렬해 반환합니다."""
    devices = glob.glob("/dev/video*")

    def sort_key(device_path):
        name = camera_name(device_path).lower()
        index = int(Path(device_path).name.removeprefix("video"))
        is_usb_camera = any(
            keyword in name for keyword in ("c920", "logitech", "usb")
        )
        return (not is_usb_camera, index)

    return sorted(devices, key=sort_key)


def open_camera(camera_source):
    """카메라를 엽니다."""
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap

    cap.release()
    return None


def open_named_camera(name_query):
    """이름으로 카메라를 찾아 엽니다."""
    for device_path in camera_candidates():
        if name_query.lower() not in camera_name(device_path).lower():
            continue

        cap = open_camera(device_path)
        if cap is not None:
            return cap, device_path

    return None, None


def find_camera():
    """사용 가능한 첫 번째 카메라를 찾아 엽니다."""
    candidates = camera_candidates()
    for device_path in candidates:
        cap = open_camera(device_path)
        if cap is not None:
            print(f"Opened {camera_name(device_path)} at {device_path}")
            return cap, device_path

    if not candidates:
        print("No /dev/video* devices are available in this environment.")

    return None, None


def resolve_camera(camera_source):
    """카메라 소스를 해석하고 카메라를 엽니다."""
    if camera_source is None:
        return find_camera()

    if camera_source.isdigit():
        device_path = f"/dev/video{camera_source}"
        return open_camera(device_path), device_path

    if os.path.exists(camera_source):
        return open_camera(camera_source), camera_source

    return open_named_camera(camera_source)


def load_roi_json(roi_json_path):
    """ROI JSON 파일을 로드합니다."""
    if not Path(roi_json_path).exists():
        raise FileNotFoundError(f"ROI JSON 파일을 찾지 못했습니다: {roi_json_path}")

    with open(roi_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def draw_roi_on_frame(frame, roi_data):
    """ROI를 프레임에 그립니다."""
    output = frame.copy()
    overlay = output.copy()

    label_colors = {
        0: (0, 165, 255),    # 하행 (주황색)
        1: (0, 255, 0),      # 상행 (녹색)
    }

    for roi_item in roi_data:
        if not isinstance(roi_item, list) or len(roi_item) != 2:
            continue

        points_data, label = roi_item
        if not isinstance(points_data, list) or len(points_data) < 3:
            continue

        # 좌표 변환
        points = np.array([[int(p[0]), int(p[1])] for p in points_data], dtype=np.int32)
        color = label_colors.get(label, (255, 255, 0))

        # ROI 폴리곤 그리기
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(output, [points], True, color, 1)

    # 반투명 오버레이 합성
    output = cv2.addWeighted(overlay, 0.15, output, 0.85, 0)

    return output


def parse_args():
    """명령어 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="USB 카메라 화면에 ROI JSON을 오버레이해 매칭을 확인합니다."
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="카메라 소스 (숫자 인덱스 또는 장치 경로 또는 이름); 비우면 첫 번째 사용 가능 카메라 사용",
    )
    parser.add_argument(
        "--roi-json",
        type=Path,
        required=True,
        help="확인할 ROI JSON 파일 경로",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="창을 띄우지 않고, ROI가 표시된 프레임 이미지만 저장합니다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="저장 이미지 폴더 경로. 기본값은 roi_check_output",
    )
    return parser.parse_args()


def print_usage():
    """사용 방법을 출력합니다."""
    print("ROI 확인 모드를 시작합니다.")
    print("사용법:")
    print("  스페이스 : 현재 프레임 저장")
    print("  q/Esc   : 종료")


def main():
    """메인 함수입니다."""
    args = parse_args()

    # ROI JSON 로드
    try:
        roi_data = load_roi_json(args.roi_json)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ROI JSON 로드 실패: {e}")
        raise SystemExit(1)

    # 카메라 열기
    cap, camera_path = resolve_camera(args.camera)
    if cap is None:
        source_description = args.camera or "any camera"
        raise SystemExit(f"Could not open camera: {source_description}")

    print(f"카메라 열기 성공: {camera_path}")
    print(f"ROI JSON 로드 성공: {args.roi_json} ({len(roi_data)} ROIs)")

    if not args.save_only:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print_usage()

    frame_count = 0
    saved_count = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 읽지 못했습니다.")
                break

            # ROI 오버레이
            display_frame = draw_roi_on_frame(frame, roi_data)
            frame_count += 1

            if args.save_only:
                # 저장 전용 모드: 첫 프레임 저장 후 종료
                output_path = args.output_dir / "roi_check_frame.jpg"
                cv2.imwrite(str(output_path), display_frame)
                print(f"프레임 저장: {output_path}")
                break
            else:
                # 대화형 모드
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    output_path = args.output_dir / f"roi_check_frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(output_path), display_frame)
                    print(f"프레임 저장: {output_path}")
                    saved_count += 1
                elif key == ord("q") or key == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"종료: {frame_count} 프레임 처리, {saved_count} 프레임 저장")


if __name__ == "__main__":
    main()
