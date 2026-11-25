# USB 카메라 발표 시연

저장소 루트에서 [공통 환경](../docs/setup.md)을 설치한 뒤 실행합니다.

```bash
python capstone_demo/main.py --roi-use --device cpu
```

주소는 `http://localhost:8000`입니다. 기본 입력은 Logitech C920이며, `--camera 0` 또는 장치 경로로 변경할 수 있습니다. 카메라가 연결돼 있어야 실시간 화면을 확인할 수 있습니다.

| 항목 | 경로 |
| --- | --- |
| 차량 모델 | `models/camera/pruned.pt`, `pruned.yaml` |
| 비교용 원본 모델 | `models/camera/raw_model.pt` |
| 사고 모델 | `models/crash/crash_pruned.pt`, `crash_pruned.yaml` |
| 사고 시연 영상 | `data/videos/crash/` |
| 카메라 ROI | `capstone_demo/referenceline/` |
| 사고 ROI·기준선 | `capstone_demo/crash_demo/referenceline/` |
| 웹 화면 | `capstone_demo/web/` |

사고 영상 모니터를 끄려면 `--no-crash-demo`를 추가합니다. 모델, ROI, 영상 폴더를 변경할 때는 `--help`의 해당 옵션을 사용합니다.

카메라 비교, ROI 확인·기준선 저장, 카메라 녹화 도구는 `utils/`에 있습니다. 전체 폴더 정리 후에도 이 앱의 내부 `utils/`, `crash_demo/`, 웹 화면 구성은 유지했습니다.
