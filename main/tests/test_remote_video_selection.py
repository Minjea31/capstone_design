"""HTTP 선택/송신기 제어를 검증합니다. WebSocket/CORS 등록은 이 테스트의 범위 밖입니다."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from utils.app_config import REMOTE_CCTV_NAMES
from utils.pipeline import PrunedCCTVPipeline


class RemoteVideoSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 실제 Flask HTTP 라우트를 사용하고 WebSocket 확장 등록만 생략합니다.
        cors_module = types.ModuleType("flask_cors")
        cors_module.CORS = lambda app: None
        sock_module = types.ModuleType("flask_sock")
        sock_module.Sock = lambda app: types.SimpleNamespace(route=lambda *args: lambda fn: fn)
        spec = importlib.util.spec_from_file_location(
            "remote_backend_for_http_tests", Path(__file__).resolve().parents[1] / "backend" / "main.py",
        )
        cls.backend = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"flask_cors": cors_module, "flask_sock": sock_module}):
            spec.loader.exec_module(cls.backend)

    def test_cctv_and_video_selection_reaches_sender_with_reordered_live_catalog(self):
        backend = self.backend
        backend.cctv_records = []
        backend.latest_detection = None
        backend.selected_index = 0
        backend.selection_version = 0
        records = [{"name": name, "cctv_url": "", "stream_url": ""} for name in REMOTE_CCTV_NAMES]
        live_df = pd.DataFrame({
            "cctvname": list(reversed(REMOTE_CCTV_NAMES)),
            "cctvurl": ["https://example.invalid/live"] * 20,
        })
        with (
            patch.object(backend, "_fetch_cctv_records", return_value=records),
            patch("utils.pipeline.fetch_cctv_dataframe", return_value=live_df),
            patch("utils.pipeline.PrunedYOLODetector"),
            patch("utils.pipeline.DETECTION_TRANSPORT", "http"),
        ):
            client = backend.app.test_client()
            pipeline = PrunedCCTVPipeline(cctv_index=0)
            pipeline._open_capture = Mock()
            try:
                pipeline.open()
                items = client.get("/api/cctvs").get_json()["items"]
                self.assertEqual(len([item for item in items if item["source_type"] == "cctv"]), 20)
                self.assertEqual([item["name"] for item in items[20:]], backend.VIDEO_ITEMS)
                for index in (20, 21, 22, 23, 0):
                    with self.subTest(index=index):
                        response = client.post(f"/api/select/{index}")
                        self.assertEqual(response.status_code, 200)
                        control = client.get("/api/control").get_json()
                        pipeline._pending_remote_control = control
                        pipeline._apply_remote_control_if_available()
                        self.assertEqual(pipeline.current_cctv_name, items[index]["name"])
                        self.assertEqual(pipeline._get_remote_display_index(), index)
                        self.assertEqual(pipeline.local_video_mode, index >= 20)
                self.assertEqual(client.post("/api/select/24").status_code, 400)
            finally:
                pipeline.release()


if __name__ == "__main__":
    unittest.main()
