from pathlib import Path
import re
from typing import Iterable, List

import cv2


MODULE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAVED_VIDEOS_DIR = MODULE_DIR.parents[1] / "data" / "videos" / "crash"
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv")


def natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_saved_videos(videos_dir: Path = DEFAULT_SAVED_VIDEOS_DIR) -> List[Path]:
    """Return saved videos sorted by filename."""
    if not videos_dir.exists():
        return []

    paths: List[Path] = []
    for suffix in VIDEO_SUFFIXES:
        paths.extend(videos_dir.glob(f"*{suffix}"))
        paths.extend(videos_dir.glob(f"*{suffix.upper()}"))
    return sorted(set(paths), key=natural_sort_key)


def resolve_video_path(requested_path: str = "") -> Path:
    """Resolve a requested video path or choose the latest saved_videos file."""
    if requested_path:
        video_path = Path(requested_path).expanduser()
        if not video_path.is_absolute():
            cwd_path = Path.cwd() / video_path
            module_path = MODULE_DIR / video_path
            video_path = cwd_path if cwd_path.exists() else module_path
        return video_path

    videos = find_saved_videos()
    if not videos:
        raise FileNotFoundError(f"saved video 파일이 없습니다: {DEFAULT_SAVED_VIDEOS_DIR}")
    return videos[-1]


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
    return capture


def get_video_delay_ms(capture: cv2.VideoCapture, default_delay: int = 30) -> int:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        return default_delay
    return max(1, min(100, int(round(1000.0 / fps))))


def read_video_frame(video_path: Path, frame_index: int = 0):
    capture = open_video_capture(video_path)
    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"프레임을 읽을 수 없습니다: {video_path}, frame={frame_index}")
        return frame
    finally:
        capture.release()


def format_video_choices(videos: Iterable[Path]) -> str:
    return "\n".join(f"- {path}" for path in videos)
