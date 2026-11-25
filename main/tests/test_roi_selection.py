import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from utils.pipeline import PrunedCCTVPipeline
from utils.roi import MAIN_DIR, build_saved_video_roi_path, load_roi_data


class ROISelectionTests(unittest.TestCase):
    def make_pipeline(self, names, **kwargs):
        cctvs = pd.DataFrame({
            "cctvname": names,
            "cctvurl": [f"https://example.invalid/{i}" for i in range(len(names))],
        })
        with (
            patch("utils.pipeline.PrunedYOLODetector"),
            patch("utils.pipeline.fetch_cctv_dataframe", return_value=cctvs),
            patch("utils.pipeline.DETECTION_TRANSPORT", "http"),
        ):
            pipeline = PrunedCCTVPipeline(cctv_index=0, use_roi=True, **kwargs)
        pipeline._open_capture = Mock()
        self.addCleanup(pipeline.release)
        return pipeline

    def test_reordered_api_list_keeps_camera_roi_on_open_and_switch(self):
        names = ["[경부선] 몽단이고개", "[경부선] 강서", "추가 CCTV"]
        for ordered_names in (names, list(reversed(names))):
            with self.subTest(names=ordered_names):
                pipeline = self.make_pipeline(ordered_names)
                pipeline.open()
                for name, filename in (
                    ("[경부선] 몽단이고개", "roi_data_15.json"),
                    ("[경부선] 강서", "roi_data_19.json"),
                ):
                    pipeline.select_cctv(ordered_names.index(name))
                    expected_path = MAIN_DIR / "roi" / filename
                    self.assertEqual(Path(pipeline.roi_path), expected_path)
                    self.assertEqual(pipeline.rois, load_roi_data(str(expected_path)))
                    self.assertTrue(pipeline.rois)

    def test_mongdan_at_api_index_19_uses_roi_15(self):
        names = [f"추가 CCTV {i}" for i in range(19)] + ["[경부선] 몽단이고개"]
        pipeline = self.make_pipeline(names)
        pipeline.cctv_index = 19
        pipeline.open()
        self.assertEqual(pipeline.current_index, 19)
        self.assertEqual(Path(pipeline.roi_path).name, "roi_data_15.json")
        self.assertTrue(pipeline.rois)

    def test_unregistered_camera_does_not_inherit_previous_camera_roi(self):
        pipeline = self.make_pipeline(["[경부선] 몽단이고개", "미등록 CCTV"])
        pipeline.open()
        self.assertTrue(pipeline.rois)
        pipeline.select_cctv(1)
        self.assertIsNone(pipeline.roi_path)
        self.assertEqual(pipeline.rois, [])

    def test_explicit_roi_override_is_preserved_on_camera_switch(self):
        custom_path = str(MAIN_DIR / "roi" / "roi_data_2.json")
        pipeline = self.make_pipeline(
            ["[경부선] 몽단이고개", "미등록 CCTV"], roi_path=custom_path,
        )
        pipeline.open()
        pipeline.select_cctv(1)
        self.assertEqual(pipeline.roi_path, custom_path)
        self.assertEqual(pipeline.rois, load_roi_data(custom_path))

    def test_failed_camera_switch_restores_previous_camera_roi(self):
        pipeline = self.make_pipeline(["[경부선] 몽단이고개", "[경부선] 강서"])
        pipeline.open()
        previous_rois = list(pipeline.rois)
        pipeline._open_capture.side_effect = RuntimeError("stream unavailable")
        with self.assertRaisesRegex(RuntimeError, "stream unavailable"):
            pipeline.select_cctv(1)
        self.assertEqual(pipeline.current_cctv_name, "[경부선] 몽단이고개")
        self.assertEqual(Path(pipeline.roi_path).name, "roi_data_15.json")
        self.assertEqual(pipeline.rois, previous_rois)

    def test_recordings_load_matching_roi_and_clear_roi_for_changed_view(self):
        pipeline = self.make_pipeline(["[경부선] 남청주육교"])
        pipeline.open()
        expected = {
            "악천후 · 2.avi": "roi_data_2.json",
            "악천후 · 3.avi": "roi_data_3.json",
            "악천후 · 6.avi": None,
            "악천후 · 10.avi": "roi_data_10.json",
        }
        for index in pipeline.video_indices:
            pipeline.select_cctv(index)
            filename = expected[pipeline.current_cctv_name]
            if filename is None:
                self.assertIsNone(pipeline.roi_path)
                self.assertEqual(pipeline.rois, [])
            else:
                path = MAIN_DIR / "roi" / filename
                self.assertEqual(Path(pipeline.roi_path), path)
                self.assertEqual(pipeline.rois, load_roi_data(str(path)))
                self.assertEqual({label for _, label in pipeline.rois}, {0, 1})
        pipeline.select_cctv(0)
        self.assertEqual(Path(pipeline.roi_path).name, "roi_data_2.json")

    def test_same_filename_in_unrelated_folder_does_not_reuse_recording_roi(self):
        self.assertIsNone(build_saved_video_roi_path("/tmp/unrelated-videos/2.avi"))


if __name__ == "__main__":
    unittest.main()
