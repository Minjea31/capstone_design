import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd

from utils.cctv_ui import VideoService
from utils.pipeline import PrunedCCTVPipeline
from utils.stream import load_video_dataframe
from utils.app_config import REMOTE_CCTV_NAMES


class LocalVideoTests(unittest.TestCase):
    def test_catalog_sorts_recordings_and_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "재생할 영상이 없습니다"):
                load_video_dataframe(directory)
            for name in ("10.avi", "2.avi", "3.MP4", "ignore.txt"):
                Path(directory, name).touch()
            catalog = load_video_dataframe(directory)
            self.assertEqual(catalog.cctvname.tolist(), ["악천후 · 2.avi", "악천후 · 3.MP4", "악천후 · 10.avi"])

    def test_local_playback_skips_api_switches_sources_and_loops_at_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in (2, 10):
                Path(directory, f"{number}.avi").touch()
            with (
                patch("utils.pipeline.PrunedYOLODetector"),
                patch("utils.pipeline.fetch_cctv_dataframe") as fetch,
                patch("utils.pipeline.connect_capture_with_retry") as connect,
                patch("utils.pipeline.open_capture") as open_capture,
                patch("utils.pipeline.DETECTION_TRANSPORT", "http"),
            ):
                first_capture = Mock()
                first_capture.get.return_value = 20.0
                second_capture = Mock()
                second_capture.get.return_value = 20.0
                open_capture.side_effect = [first_capture, second_capture]
                pipeline = PrunedCCTVPipeline(video_dir=directory)
                try:
                    pipeline.open()
                    self.assertEqual(pipeline.current_cctv_name, "악천후 · 2.avi")
                    fetch.assert_not_called()
                    connect.assert_not_called()
                    service = VideoService.__new__(VideoService)
                    service.pipeline = pipeline
                    items = service._build_visible_cctv_items()
                    self.assertEqual(len([item for item in items if item['source_type'] == 'cctv']), 20)
                    self.assertEqual([item['name'] for item in items if item['source_type'] == 'video'], ['악천후 · 2.avi', '악천후 · 10.avi'])

                    frame = np.zeros((32, 32, 3), dtype=np.uint8)
                    first_capture.read.side_effect = [(False, None), (True, frame)]
                    pipeline.detector.reset_tracking.reset_mock()
                    ok, result = pipeline._read_frame()
                    self.assertTrue(ok)
                    self.assertIs(result, frame)
                    first_capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
                    pipeline.detector.reset_tracking.assert_called_once()

                    pipeline.select_cctv(pipeline.video_indices[1])
                    self.assertEqual(pipeline.current_cctv_name, "악천후 · 10.avi")
                    first_capture.release.assert_called_once()
                    self.assertIs(pipeline.capture, second_capture)
                finally:
                    pipeline.release()

    def test_switching_back_to_live_fetches_urls_and_uses_network_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "2.avi").touch()
            live = pd.DataFrame({"cctvname": REMOTE_CCTV_NAMES, "cctvurl": [f"https://example.invalid/{i}" for i in range(20)]})
            with (
                patch("utils.pipeline.PrunedYOLODetector"),
                patch("utils.pipeline.fetch_cctv_dataframe", return_value=live) as fetch,
                patch("utils.pipeline.connect_capture_with_retry", return_value=("stream", Mock())) as connect,
                patch("utils.pipeline.open_capture", return_value=Mock(get=Mock(return_value=20.0))),
                patch("utils.pipeline.DETECTION_TRANSPORT", "http"),
            ):
                pipeline = PrunedCCTVPipeline(video_dir=directory)
                try:
                    pipeline.open()
                    self.assertTrue(pipeline.local_video_mode)
                    pipeline.select_cctv(0)
                    self.assertFalse(pipeline.local_video_mode)
                    self.assertEqual(pipeline.current_cctv_name, REMOTE_CCTV_NAMES[0])
                    fetch.assert_called_once()
                    connect.assert_called_once_with("https://example.invalid/0")
                    pipeline.select_cctv(pipeline.video_indices[0])
                    self.assertTrue(pipeline.local_video_mode)
                    self.assertEqual(pipeline._get_remote_display_index(), 20)
                    previous_capture = pipeline.capture
                    pipeline.cctv_df.at[1, 'cctvurl'] = ''
                    fetch.side_effect = OSError('API unavailable')
                    with self.assertRaisesRegex(RuntimeError, 'CCTV 목록 조회 실패'):
                        pipeline.select_cctv(1)
                    self.assertTrue(pipeline.local_video_mode)
                    self.assertIs(pipeline.capture, previous_capture)
                    self.assertEqual(pipeline.current_cctv_name, '악천후 · 2.avi')
                finally:
                    pipeline.release()


if __name__ == "__main__":
    unittest.main()
