# 작업 도구

저장소 루트에서 `python tools/<스크립트>.py`로 실행합니다. 공통 환경은 [설치 문서](../docs/setup.md)를 참고합니다.

| 파일 | 용도 |
| --- | --- |
| `verify_runtime.py` | 실제 저장 영상·모델·추적·ROI·웹 응답 검증 |
| `compare_saved_video_detect.py` | 원본 모델과 최종 모델 비교 |
| `compare_saved_video_detect_g040.py` | 원본 모델과 GA g040 모델 비교 |
| `record_screen_region.py` | 모델 비교 패널을 영상으로 저장 |
| `live_opencv.py`, `yolo_opencv.py` | 실시간 CCTV 수신·검출 확인 |
| `video_save.py`, `frame_save.py` | CCTV 영상·프레임 저장 |
| `roi.py`, `roi_check.py` | CCTV ROI 작성·확인 |
| `save_reference_lines.py` | ROI 기준선 작성 |
| `find.py` | ITS CCTV 검색 |
| `check_dataset.py` | 라벨·이미지 데이터셋 점검 |

```bash
python tools/compare_saved_video_detect_g040.py --device cpu
python tools/record_screen_region.py --help
python tools/roi_check.py --index 2
```

모델은 `models/experiments`, 기본 영상은 `data/videos/weather`에서 읽습니다. ROI 기본 경로는 `main/roi`입니다. 추가 녹화와 비교 결과는 `outputs/` 같은 Git 제외 경로에 저장하세요.
