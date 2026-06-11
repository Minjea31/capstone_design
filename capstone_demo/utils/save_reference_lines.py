import argparse
import json
import glob
import os
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Save Heading Reference Groups"
MODULE_DIR = Path(__file__).resolve().parent
POINT_SELECT_RADIUS = 18
GROUP_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]
ROI_LABEL_COLORS = {
    0: (0, 165, 255),
    1: (0, 255, 0),
}

STATE = {
    "display_frame": None,
    "roi_points": [],
    "groups": [],
    "current_group": [],
}


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


def get_first_frame_from_camera(camera_source=None):
    """카메라에서 첫 프레임을 가져옵니다."""
    cap, camera_path = resolve_camera(camera_source)

    if cap is None:
        source_description = camera_source or "any camera"
        raise SystemExit(
            f"Could not open camera: {source_description}"
        )

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Could not read frame from camera: {camera_path}")

    return frame, cap, camera_path


def resolve_roi_json_path(requested_path, save_index):
    """ROI JSON 경로를 해석합니다."""
    if requested_path:
        return Path(requested_path)
    referenceline_dir = MODULE_DIR.parent / "referenceline"
    referenceline_dir.mkdir(parents=True, exist_ok=True)
    return referenceline_dir / f"roi_data_{int(save_index)}.json"


def resolve_output_json_path(requested_path, save_index):
    """출력 JSON 경로를 해석합니다."""
    if requested_path:
        return Path(requested_path)
    referenceline_dir = MODULE_DIR.parent / "referenceline"
    referenceline_dir.mkdir(parents=True, exist_ok=True)
    return referenceline_dir / f"heading_{int(save_index)}.json"


def load_roi_points(roi_json_path):
    """ROI JSON 파일을 로드하고 점 리스트를 생성합니다."""
    if not roi_json_path.exists():
        raise FileNotFoundError(f"ROI JSON 파일을 찾지 못했습니다: {roi_json_path}")

    with roi_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    roi_points = []
    point_id = 0
    for roi_index, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            continue
        points_data, label = item
        if not isinstance(points_data, list):
            continue

        normalized_points = []
        for point_index, point in enumerate(points_data):
            if not isinstance(point, list) or len(point) != 2:
                continue
            x = int(point[0])
            y = int(point[1])
            normalized_points.append((x, y))
            roi_points.append(
                {
                    "id": point_id,
                    "roi_index": int(roi_index),
                    "roi_label": int(label),
                    "point_index": int(point_index),
                    "point": (x, y),
                }
            )
            point_id += 1

    if not roi_points:
        raise RuntimeError(f"선택 가능한 ROI 점이 없습니다: {roi_json_path}")

    return roi_points


def build_save_payload(index, name, roi_json_path, groups):
    """저장할 payload를 구성합니다."""
    serializable_groups = []
    for group_index, group in enumerate(groups):
        points = [[int(item["point"][0]), int(item["point"][1])] for item in group]
        selected_ids = [int(item["id"]) for item in group]
        roi_indexes = [int(item["roi_index"]) for item in group]
        roi_labels = [int(item["roi_label"]) for item in group]
        serializable_groups.append(
            {
                "group_id": int(group_index),
                "point_ids": selected_ids,
                "points": points,
                "roi_indexes": roi_indexes,
                "roi_labels": roi_labels,
            }
        )

    return {
        "index": int(index),
        "name": name or f"Camera {index}",
        "source_roi_json": str(roi_json_path),
        "groups": serializable_groups,
    }


def save_group_json(output_json_path, index, name, roi_json_path, groups):
    """그룹 JSON을 저장합니다."""
    payload = build_save_payload(index, name, roi_json_path, groups)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_nearest_roi_point(x, y, roi_points, excluded_ids=None):
    """마우스 클릭 위치 근처의 ROI 점을 찾습니다."""
    excluded_ids = excluded_ids or set()
    best_item = None
    best_distance_sq = None

    for item in roi_points:
        if item["id"] in excluded_ids:
            continue
        px, py = item["point"]
        distance_sq = (px - x) ** 2 + (py - y) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_item = item

    if best_item is None:
        return None

    if best_distance_sq is None or best_distance_sq > POINT_SELECT_RADIUS * POINT_SELECT_RADIUS:
        return None

    return best_item


def draw_group(overlay, group, color, thickness):
    """그룹을 화면에 그립니다."""
    if len(group) >= 2:
        cv2.polylines(
            overlay,
            [np.array([item["point"] for item in group], dtype=np.int32)],
            False,
            color,
            thickness,
        )

    for item in group:
        x, y = item["point"]
        cv2.circle(overlay, (x, y), 7, color, -1)
        cv2.circle(overlay, (x, y), 10, (255, 255, 255), 1)


def draw_overlay(frame, roi_points, groups, current_group):
    """ROI와 그룹을 프레임에 오버레이로 그립니다."""
    overlay = frame.copy()

    points_by_roi = {}
    for item in roi_points:
        points_by_roi.setdefault(item["roi_index"], []).append(item)

    for items in points_by_roi.values():
        polygon = np.array([item["point"] for item in items], dtype=np.int32)
        roi_label = items[0]["roi_label"]
        color = ROI_LABEL_COLORS.get(roi_label, (255, 255, 255))
        if len(items) >= 2:
            cv2.polylines(overlay, [polygon], True, color, 1)

    selected_ids = {item["id"] for group in groups for item in group}
    selected_ids.update(item["id"] for item in current_group)

    for item in roi_points:
        x, y = item["point"]
        roi_label = item["roi_label"]
        point_color = ROI_LABEL_COLORS.get(roi_label, (220, 220, 220))
        radius = 5 if item["id"] not in selected_ids else 3
        cv2.circle(overlay, (x, y), radius, point_color, -1)

    for group_index, group in enumerate(groups):
        color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
        draw_group(overlay, group, color, 2)

    active_group_index = len(groups)
    active_color = GROUP_COLORS[active_group_index % len(GROUP_COLORS)]
    draw_group(overlay, current_group, active_color, 3)

    return overlay


def mouse_callback(event, x, y, flags, param):
    """마우스 콜백 함수입니다."""
    del flags, param
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    current_ids = {item["id"] for item in STATE["current_group"]}
    nearest_point = get_nearest_roi_point(x, y, STATE["roi_points"], excluded_ids=current_ids)
    if nearest_point is None:
        print(f"선택 가능한 ROI 점이 없습니다. 클릭 위치=({x}, {y})")
        return

    STATE["current_group"].append(nearest_point)
    print(
        "선택 점:",
        f"id={nearest_point['id']}",
        f"roi={nearest_point['roi_index']}",
        f"label={nearest_point['roi_label']}",
        f"point_index={nearest_point['point_index']}",
        f"point={nearest_point['point']}",
    )


def parse_args():
    """명령어 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="USB 카메라에서 ROI 점을 선택하고 grouped heading reference JSON을 저장합니다"
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="카메라 소스 (숫자 인덱스 또는 장치 경로 또는 이름); 비우면 첫 번째 사용 가능 카메라 사용",
    )
    parser.add_argument(
        "--roi-json",
        default="",
        help="입력 ROI JSON 경로; 비우면 roi_data_<index>.json",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="출력 JSON 경로; 비우면 heading_<index>.json",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="저장할 카메라 index; 기본값은 0",
    )
    parser.add_argument(
        "--name",
        default="",
        help="저장할 카메라 name; 기본값은 Camera <index>",
    )
    return parser.parse_args()


def print_usage(index, name, source_label, roi_json_path, output_json_path, roi_points):
    """사용 방법을 출력합니다."""
    print("reference line 작업을 시작합니다. 입력은 USB 카메라 스트림입니다.")
    print(f"index: {index}")
    print(f"name: {name or f'Camera {index}'}")
    print(f"camera: {source_label}")
    print(f"ROI JSON: {roi_json_path}")
    print(f"저장 경로: {output_json_path}")
    print(f"선택 가능한 ROI 점 개수: {len(roi_points)}")
    print("사용법:")
    print("  좌클릭 : 가까운 ROI 점 선택")
    print("  n      : 현재 선택한 점들을 reference line 그룹으로 확정")
    print("  u      : 현재 그룹에서 마지막 점 취소")
    print("  d      : 마지막으로 확정한 그룹 삭제")
    print("  r      : 모든 그룹/현재 선택 초기화")
    print(f"  s      : {output_json_path} 저장")
    print("  q/Esc  : 종료")
    print("권장 순서: 하행 line 1 -> 하행 line 2 -> 상행 line 1 -> 상행 line 2")
    print("점 클릭 방향: 하행은 위에서 아래, 상행은 아래에서 위")


def main():
    """메인 함수입니다."""
    args = parse_args()

    frame, capture, camera_path = get_first_frame_from_camera(args.camera)
    save_index = args.index if args.index is not None else 0
    save_name = args.name or f"Camera {save_index}"
    source_label = f"USB Camera {camera_path}"

    roi_json_path = resolve_roi_json_path(args.roi_json, save_index)
    output_json_path = resolve_output_json_path(args.output_json, save_index)

    try:
        STATE["display_frame"] = frame.copy()
        STATE["roi_points"] = load_roi_points(roi_json_path)
        STATE["groups"] = []
        STATE["current_group"] = []

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
        print_usage(save_index, save_name, source_label, roi_json_path, output_json_path, STATE["roi_points"])

        while True:
            overlay = draw_overlay(
                STATE["display_frame"],
                STATE["roi_points"],
                STATE["groups"],
                STATE["current_group"],
            )
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("u"):
                if STATE["current_group"]:
                    removed = STATE["current_group"].pop()
                    print(f"현재 그룹에서 점 제거: id={removed['id']} point={removed['point']}")
            elif key == ord("n"):
                if len(STATE["current_group"]) < 2:
                    print("그룹을 확정하려면 현재 그룹에 최소 2개 점이 필요합니다.")
                    continue
                STATE["groups"].append(list(STATE["current_group"]))
                print(f"그룹 확정 완료: group_id={len(STATE['groups']) - 1}, points={len(STATE['current_group'])}")
                STATE["current_group"] = []
            elif key == ord("d"):
                if STATE["groups"]:
                    removed_group = STATE["groups"].pop()
                    print(f"마지막 저장 그룹 삭제: points={len(removed_group)}")
            elif key == ord("r"):
                STATE["groups"] = []
                STATE["current_group"] = []
                print("모든 그룹과 현재 선택을 초기화했습니다.")
            elif key == ord("s"):
                groups_to_save = list(STATE["groups"])
                if len(STATE["current_group"]) >= 2:
                    groups_to_save.append(list(STATE["current_group"]))

                if not groups_to_save:
                    print("저장하려면 최소 한 개 그룹에 2개 이상 점이 필요합니다.")
                    continue

                save_group_json(output_json_path, save_index, save_name, roi_json_path, groups_to_save)
                print(f"저장 완료: {output_json_path} (groups={len(groups_to_save)})")
            elif key == ord("q") or key == 27:
                break
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
