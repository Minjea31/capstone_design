#!/usr/bin/env python3
"""Check real model inference and local HTTP routes without cameras or ITS access.

Run from any directory. Use --report and --compare to check later refactors
against a report produced with the same CPU environment and input files.
"""

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    # This command must not send frames or control requests to a remote server.
    for key in list(os.environ):
        if key.startswith("DETECTION_"):
            del os.environ[key]
    sys.path[:0] = [str(ROOT / "main"), str(ROOT / "capstone_demo")]

    import cv2
    import torch
    from utils import app_config
    from utils.cctv_ui import create_app
    from utils.detector import load_model
    from utils.pipeline import PrunedCCTVPipeline
    from crash_demo import detector as crash

    torch.set_num_threads(1)
    # Stable comparisons also work on machines with a CUDA device.
    app_config.INFERENCE_DEVICE = "cpu"
    with patch("utils.detector.INFERENCE_DEVICE", "cpu"):
        cctv_model = load_model()
    camera = import_file("camera_verification", ROOT / "capstone_demo" / "main.py")
    with patch.object(sys, "argv", ["main.py"]):
        camera.args = camera.parse_args()

    weather_dir = app_config.SAVED_VIDEOS_DIR
    crash_dir = crash.DEFAULT_SAVED_VIDEOS_DIR
    weather = [weather_dir / f"{number}.avi" for number in (2, 3, 6, 10)]
    crash_videos = [crash_dir / f"{number}.avi" for number in (1, 2, 3)]
    report = {"inference": {}, "camera_http": {}, "pipeline": [], "local_http": {}, "backend_http": {}}
    camera_model = camera.load_model(camera.DEFAULT_PRUNED_MODEL_PATH, camera.DEFAULT_MODEL_YAML_PATH, "cpu")
    crash_model = crash._load_yolo_model(crash.DEFAULT_WEIGHTS_PATH, crash.DEFAULT_MODEL_YAML_PATH, "cpu")
    for name, model, videos, confidence in (
        ("cctv", cctv_model, weather, 0.4),
        ("camera", camera_model, weather[:1], 0.4),
        ("crash", crash_model, crash_videos, 0.7),
    ):
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            try:
                for index in range(3):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot read demo video: {video}. Run git lfs pull.")
                    prediction = model.predict(frame, device="cpu", classes=[0], conf=confidence, verbose=False)[0]
                    report["inference"][f"{name}/{video.name}/{index}"] = {
                        "frame_sha256": sha256(frame.tobytes()),
                        "boxes": prediction.boxes.data.cpu().numpy().tolist(),
                    }
            finally:
                capture.release()

    client = camera.app.test_client()
    for route in ("/", "/script.js", "/style.css", "/api/status", "/api/cctvs", "/crash_video_feed"):
        response = client.get(route)
        expected = 404 if route == "/crash_video_feed" else 200
        if response.status_code != expected:
            raise RuntimeError(f"Camera route failed: {route}: {response.status_code}")
        report["camera_http"][route] = {"status": response.status_code, "sha256": sha256(response.data)}

    with patch("utils.detector.INFERENCE_DEVICE", "cpu"):
        pipeline = PrunedCCTVPipeline(video_dir=str(weather_dir), use_roi=True)
        try:
            pipeline.open()
            for _ in range(3):
                result = pipeline.process_next_frame()
                if not result["roi_enabled"] or result["frame"].size == 0:
                    raise RuntimeError("ROI/rendering check failed")
                report["pipeline"].append({
                    "frame_sha256": sha256(result["frame"].tobytes()),
                    "traffic_count": result["traffic_count"],
                    "traffic_up": result["traffic_up"],
                    "traffic_down": result["traffic_down"],
                })
        finally:
            pipeline.release()

        # Real Flask service, capture and model; no mocked UI/background worker.
        web_pipeline = PrunedCCTVPipeline(video_dir=str(weather_dir), use_roi=True)
        app = create_app(web_pipeline)
        service = app.config["VIDEO_SERVICE"]
        try:
            client = app.test_client()
            for route in ("/", "/static/style.css", "/api/status", "/api/cctvs"):
                response = client.get(route)
                if response.status_code != 200:
                    raise RuntimeError(f"Local route failed: {route}")
                report["local_http"][route] = response.status_code
            catalog = client.get("/api/cctvs").get_json()["items"]
            if len(catalog) != 24 or sum(item["source_type"] == "video" for item in catalog) != 4:
                raise RuntimeError("Expected 20 CCTV entries and four weather videos")
        finally:
            service.release()
            atexit.unregister(service.release)

    backend = import_file("backend_verification", ROOT / "main" / "backend" / "main.py")
    records = [{"name": name, "cctv_url": "", "stream_url": ""} for name in app_config.REMOTE_CCTV_NAMES]
    # Supply the external ITS response; use the actual CORS/WebSocket extensions.
    with patch.object(backend, "_fetch_cctv_records", return_value=records):
        client = backend.app.test_client()
        for route in ("/health", "/api/cctvs", "/api/control", "/api/status", "/api/latest"):
            response = client.get(route)
            if response.status_code != 200:
                raise RuntimeError(f"Backend route failed: {route}")
            report["backend_http"][route] = response.status_code
        if client.post("/api/select/20").status_code != 200:
            raise RuntimeError("Remote video selection failed")
        if client.get("/api/control").get_json()["selected_name"] != "악천후 · 2.avi":
            raise RuntimeError("Remote selected video did not reach control endpoint")
        if client.post("/api/select/24").status_code != 400:
            raise RuntimeError("Invalid selection was not rejected")
        response = client.post("/api/detection", json={
            "selected_name": "악천후 · 2.avi",
            "detections": [
                {"bbox": [0, 0, 10, 10], "confidence": 0.9, "direction_label": 0},
                {"bbox": [20, 20, 30, 30], "confidence": 0.8, "direction_label": 1},
            ],
        })
        if response.status_code != 200 or client.get("/api/latest").get_json()["data"]["traffic_count"] != 2:
            raise RuntimeError("Detection receiver failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write JSON results, for example outputs/runtime.json")
    parser.add_argument("--compare", type=Path, help="Compare with an earlier JSON report")
    args = parser.parse_args()
    report = verify()
    if args.compare and json.loads(args.compare.read_text(encoding="utf-8")) != report:
        raise SystemExit("Runtime results differ from the supplied report")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(report['inference'])} inference frames; tracking/ROI/rendering; 15 HTTP routes; backend selection/ingest")
    if args.compare:
        print("PASS: results match the supplied report")


if __name__ == "__main__":
    main()
