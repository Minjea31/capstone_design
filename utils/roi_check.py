from pathlib import Path
import argparse

import cv2

from live_opencv import (
    CCTV_INDEX as DEFAULT_CCTV_INDEX,
    DIRECT_CCTV_NAME,
    DIRECT_CCTV_URL,
    FRAME_WAIT_TIMEOUT_SECONDS,
    fetch_cctv_dataframe,
    open_capture,
    read_frame_with_timeout,
    resolve_stream_url,
    select_cctv,
)
from roi import ROISelector


ROI_DIR = Path("roi")
OUTPUT_DIR = Path("roi_check_output")
WINDOW_NAME = "ROI Check"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCTV 화면에 ROI JSON을 오버레이해 매칭만 확인합니다."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_CCTV_INDEX,
        help="확인할 CCTV 인덱스. 기본값은 live_opencv.py의 CCTV_INDEX입니다.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="창을 띄우지 않고, ROI가 표시된 프레임 이미지만 저장합니다.",
    )
    return parser.parse_args()


def open_live_capture_for_index(cctv_index: int):
    if DIRECT_CCTV_URL:
        cctv_name = DIRECT_CCTV_NAME or "직접 지정 CCTV"
        cctv_url = DIRECT_CCTV_URL
    else:
        import live_opencv

        original_index = live_opencv.CCTV_INDEX
        try:
            live_opencv.CCTV_INDEX = cctv_index
            cctv_df = fetch_cctv_dataframe()
            cctv_name, cctv_url = select_cctv(cctv_df)
        finally:
            live_opencv.CCTV_INDEX = original_index

    print(f"확인할 CCTV_INDEX: {cctv_index}")
    print("선택 CCTV:", cctv_name)
    print("cctvurl:", cctv_url)

    stream_url = resolve_stream_url(cctv_url)
    print("stream url:", stream_url)

    capture = open_capture(stream_url)
    if capture is None:
        print("실제 스트림 URL 연결 실패, 원본 cctvurl로 다시 시도합니다.")
        capture = open_capture(cctv_url)

    if capture is None:
        raise RuntimeError("OpenCV에서 CCTV 스트림을 열지 못했습니다.")

    return capture, cctv_name


def annotate_frame(frame, roi_selector: ROISelector, cctv_index: int, roi_path: Path, cctv_name: str):
    display_frame = roi_selector.draw(frame)
    cv2.putText(
        display_frame,
        f"CCTV_INDEX: {cctv_index}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"ROI File: {roi_path.name}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"CCTV: {cctv_name}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display_frame


def save_preview_image(display_frame, cctv_index: int, roi_path: Path) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"roi_check_{cctv_index}.jpg"
    cv2.imwrite(str(output_path), display_frame)
    print(f"확인용 이미지를 저장했습니다: {output_path}")
    print(f"매칭 확인: CCTV_INDEX={cctv_index} <-> {roi_path}")
    return output_path


def run_save_only(capture, roi_selector: ROISelector, cctv_index: int, roi_path: Path, cctv_name: str):
    ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
    if not ok:
        raise RuntimeError(
            f"{FRAME_WAIT_TIMEOUT_SECONDS}초 동안 프레임을 받지 못해 이미지를 저장할 수 없습니다."
        )

    display_frame = annotate_frame(frame, roi_selector, cctv_index, roi_path, cctv_name)
    save_preview_image(display_frame, cctv_index, roi_path)


def run_window_mode(capture, roi_selector: ROISelector, cctv_index: int, roi_path: Path, cctv_name: str):
    cv2.namedWindow(WINDOW_NAME)

    try:
        while capture.isOpened():
            ok, frame = read_frame_with_timeout(capture, FRAME_WAIT_TIMEOUT_SECONDS)
            if not ok:
                print(
                    f"[프레임 수신 불가] {FRAME_WAIT_TIMEOUT_SECONDS}초 동안 기다렸지만 프레임이 오지 않아 종료합니다"
                )
                break

            display_frame = annotate_frame(frame, roi_selector, cctv_index, roi_path, cctv_name)
            cv2.putText(
                display_frame,
                "Press q or ESC to exit",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.index is None:
        raise RuntimeError("CCTV 인덱스를 알 수 없습니다. --index 옵션으로 지정해 주세요.")

    roi_path = ROI_DIR / f"roi_data_{args.index}.json"
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI JSON 파일이 없습니다: {roi_path}")

    print(f"매칭 확인 준비: CCTV_INDEX={args.index} <-> {roi_path}")
    roi_selector = ROISelector(window_name=WINDOW_NAME, save_path=str(roi_path))
    capture, cctv_name = open_live_capture_for_index(args.index)

    try:
        if args.save_only:
            run_save_only(capture, roi_selector, args.index, roi_path, cctv_name)
            return

        try:
            run_window_mode(capture, roi_selector, args.index, roi_path, cctv_name)
        except cv2.error as exc:
            print("OpenCV GUI를 사용할 수 없어 이미지 저장 모드로 전환합니다.")
            print(f"원인: {exc}")
            run_save_only(capture, roi_selector, args.index, roi_path, cctv_name)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
