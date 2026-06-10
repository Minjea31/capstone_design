import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
ROIData = Tuple[List[Point], int]


class ROISelector:
    """프레임 위에서 다중 ROI를 정의하고 시각화하는 도우미 클래스."""

    def __init__(
        self,
        window_name: str = "frame",
        save_path: Optional[str] = None,
        default_label: int = 0,
    ) -> None:
        # ROI를 그릴 OpenCV 창 이름과 저장 경로를 초기화합니다.
        self.window_name = window_name
        self.save_path = save_path or self._default_save_path()
        self.current_label = default_label

        self.current_points: List[Point] = []
        self.rois: List[ROIData] = []

        self._latest_frame: Optional[np.ndarray] = None
        self._mouse_callback_registered = False

        self.label_colors = {
            0: (0, 165, 255),
            1: (0, 255, 0),
        }

        self.load()

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """마우스 좌클릭으로 현재 다각형의 점을 추가합니다."""
        del flags, param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((int(x), int(y)))

    def handle_key(self, key: int) -> None:
        """키 입력을 처리해 라벨 변경, ROI 확정, 초기화, 저장을 수행합니다."""
        if key < 0:
            return

        if key == ord("0"):
            self.current_label = 0
            print("현재 ROI 라벨: 0 (하행)")
        elif key == ord("1"):
            self.current_label = 1
            print("현재 ROI 라벨: 1 (상행)")
        elif key == ord("c"):
            self.current_points.clear()
            print("현재 찍던 ROI 점을 초기화했습니다.")
        elif key == ord("n"):
            self._finalize_current_roi()
        elif key == ord("s"):
            self.save()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """현재 점과 확정된 ROI를 프레임 위에 시각화합니다."""
        output = frame.copy()
        overlay = output.copy()

        for points, label in self.rois:
            if len(points) < 3:
                continue

            polygon = np.array(points, dtype=np.int32)
            color = self.label_colors.get(label, (255, 255, 0))

            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(output, [polygon], True, color, 2)

        # 반투명 ROI 영역을 원본 위에 합성합니다.
        output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

        if self.current_points:
            current_polygon = np.array(self.current_points, dtype=np.int32)
            color = self.label_colors.get(self.current_label, (255, 255, 0))

            for point in self.current_points:
                cv2.circle(output, point, 4, (0, 0, 255), -1)

            if len(self.current_points) >= 2:
                cv2.polylines(output, [current_polygon], False, color, 2)

        return output

    def save(self) -> None:
        """확정된 ROI 목록을 JSON 파일 형식으로 저장합니다."""
        serializable_data = [
            [[[int(x), int(y)] for x, y in points], int(label)]
            for points, label in self.rois
        ]

        with open(self.save_path, "w", encoding="utf-8") as file:
            json.dump(serializable_data, file, ensure_ascii=False, indent=2)

        print(f"ROI data saved to {self.save_path}")

    def load(self) -> None:
        """저장된 ROI JSON 파일이 있으면 메모리로 불러옵니다."""
        save_file = Path(self.save_path)
        if not save_file.exists():
            return

        with save_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        loaded_rois: List[ROIData] = []
        for item in payload:
            if not isinstance(item, list) or len(item) != 2:
                continue

            points_data, label = item
            if not isinstance(points_data, list):
                continue

            points: List[Point] = []
            for point in points_data:
                if not isinstance(point, list) or len(point) != 2:
                    continue
                points.append((int(point[0]), int(point[1])))

            if len(points) >= 3:
                loaded_rois.append((points, int(label)))

        self.rois = loaded_rois
        print(f"Loaded {len(self.rois)} ROI(s) from {self.save_path}")

    def update(self, frame: np.ndarray) -> np.ndarray:
        """외부에서 전달된 프레임을 받아 ROI가 그려진 프레임을 반환합니다."""
        self._latest_frame = frame
        self._ensure_mouse_callback()
        return self.draw(frame)

    def _finalize_current_roi(self) -> None:
        """현재 점 목록을 하나의 ROI로 확정합니다."""
        if len(self.current_points) < 3:
            print("ROI를 확정하려면 최소 3개의 점이 필요합니다.")
            return

        roi_points = [(int(x), int(y)) for x, y in self.current_points]
        self.rois.append((roi_points, int(self.current_label)))
        print(f"ROI 확정 완료: label={self.current_label}, points={len(roi_points)}, total={len(self.rois)}")
        self.current_points.clear()

    def _ensure_mouse_callback(self) -> None:
        """OpenCV 창에 마우스 콜백을 연결합니다."""
        if self._mouse_callback_registered:
            return

        try:
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self._mouse_callback_registered = True
        except cv2.error:
            # 창이 아직 생성되지 않았다면 다음 update 호출에서 다시 시도합니다.
            self._mouse_callback_registered = False

    def _default_save_path(self) -> str:
        """capstone_demo/referenceline 폴더에 기본 저장 경로를 생성합니다."""
        try:
            from live_opencv import CCTV_INDEX
            index = int(CCTV_INDEX)
        except (ImportError, TypeError, ValueError):
            index = 0
        
        referenceline_dir = Path(__file__).resolve().parent.parent / "referenceline"
        referenceline_dir.mkdir(parents=True, exist_ok=True)
        return str(referenceline_dir / f"roi_data_{index}.json")


def _camera_name(device_path):
    """카메라 장치의 이름을 가져옵니다."""
    device_name = Path(os.path.realpath(device_path)).name
    name_path = Path("/sys/class/video4linux") / device_name / "name"

    try:
        return name_path.read_text().strip()
    except OSError:
        return device_name


def _camera_candidates():
    """사용 가능한 카메라 장치 목록을 우선순위 순으로 정렬해 반환합니다."""
    devices = glob.glob("/dev/video*")

    def sort_key(device_path):
        name = _camera_name(device_path).lower()
        index = int(Path(device_path).name.removeprefix("video"))
        is_usb_camera = any(
            keyword in name for keyword in ("c920", "logitech", "usb")
        )
        return (not is_usb_camera, index)

    return sorted(devices, key=sort_key)


def _open_camera(camera_source):
    """카메라를 엽니다."""
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap

    cap.release()
    return None


def _open_named_camera(name_query):
    """이름으로 카메라를 찾아 엽니다."""
    for device_path in _camera_candidates():
        if name_query.lower() not in _camera_name(device_path).lower():
            continue

        cap = _open_camera(device_path)
        if cap is not None:
            return cap, device_path

    return None, None


def _find_camera():
    """사용 가능한 첫 번째 카메라를 찾아 엽니다."""
    candidates = _camera_candidates()
    for device_path in candidates:
        cap = _open_camera(device_path)
        if cap is not None:
            print(f"Opened {_camera_name(device_path)} at {device_path}")
            return cap, device_path

    if not candidates:
        print("No /dev/video* devices are available in this environment.")

    return None, None


def _resolve_camera(camera_source):
    """카메라 소스를 해석하고 카메라를 엽니다."""
    if camera_source is None:
        return _find_camera()

    if camera_source.isdigit():
        device_path = f"/dev/video{camera_source}"
        return _open_camera(device_path), device_path

    if os.path.exists(camera_source):
        return _open_camera(camera_source), camera_source

    return _open_named_camera(camera_source)


def _open_live_capture():
    """USB 카메라를 열어 ROI 설정용 스트림을 반환합니다."""
    cap, camera_path = _resolve_camera(None)

    if cap is None:
        raise RuntimeError("OpenCV에서 USB 카메라를 열지 못했습니다.")

    print(f"카메라 열기 성공: {camera_path}")
    return cap


def print_usage(save_path: str) -> None:
    print("ROI 작업을 시작합니다. 입력은 USB 카메라 스트림입니다.")
    print(f"저장 경로: {save_path}")
    print("사용법:")
    print("  좌클릭 : ROI polygon 점 추가")
    print("  0      : 현재 ROI 라벨을 하행으로 설정")
    print("  1      : 현재 ROI 라벨을 상행으로 설정")
    print("  n      : 현재 ROI 확정")
    print("  c      : 현재 찍던 점들 초기화")
    print(f"  s      : {save_path} 저장")
    print("  q/Esc  : 종료")
    print("현재 기본 라벨: 0 (하행)")


def main() -> None:
    """ROI 선택기만 단독 실행할 수 있도록 USB 카메라 창을 엽니다."""
    window_name = "ROI Selector"
    roi_selector = ROISelector(window_name=window_name)
    capture = _open_live_capture()

    cv2.namedWindow(window_name)
    print_usage(roi_selector.save_path)

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                print("카메라에서 프레임을 읽지 못했습니다.")
                break

            display_frame = roi_selector.update(frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF
            roi_selector.handle_key(key)

            if key == ord("q") or key == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
