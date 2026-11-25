# 사고 감지 실험

`crash_test/`는 저장 영상을 대상으로 사고 후보, 이동 방향, 기준선, 선택적 깊이 추정을 실험하는 코드입니다. 발표용 웹 앱은 `capstone_demo/`에 있습니다.

```bash
# 저장소 루트에서
python experiments/crash_test/yolo_detect.py --video data/videos/crash/1.avi
python experiments/crash_test/crash_detect.py --help
python experiments/crash_test/crash_depth.py --help
```

기본 모델은 `models/crash/crash_pruned.pt`와 `crash_pruned.yaml`, 영상은 `data/videos/crash/`입니다. ROI와 기준선은 `crash_test/referenceline/`에 보존했습니다. `utils/`는 이 실험의 내부 모듈·ROI 편집 도구입니다.

OpenCV 표시 창은 GUI가 있는 환경에서 실행합니다. Depth Anything 실험은 `transformers` 및 추가 모델 다운로드가 필요합니다. 공통 설치는 [환경 구축 문서](../docs/setup.md)를 참고하세요.
