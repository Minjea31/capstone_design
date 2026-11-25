# 폴더 정리 후 동작 검증

검증일: 2026-11-25. Linux x86_64, Python 3.10, PyTorch 2.2.2, OpenCV 4.9.0, Node.js 22 환경에서 확인했습니다.

## 정리 전후 비교

정리 시작 당시에는 `ultralytics`가 삭제된 `_deps/yolov12`에 editable 설치돼 있어 테스트를 불러올 수 없었습니다. YOLOv12를 일반 패키지로 재설치한 뒤, 경로 변경 전에 기준 결과를 만들고 동일한 환경에서 비교했습니다. 사용한 소스 커밋은 `requirements.txt`에 고정했습니다.

| 점검 | 결과 |
| --- | --- |
| 모델·영상·ROI·웹 리소스 78개 SHA-256 비교 | 이동 전후 바이트 단위 동일 |
| CCTV 모델: 악천후 영상 4개 × 첫 3프레임 | 프레임 해시·검출 좌표·클래스·신뢰도 동일 |
| USB 시연 모델: 악천후 영상 1개 × 첫 3프레임 | 프레임 해시·검출 결과 동일 |
| 사고 모델: 사고 영상 3개 × 첫 3프레임 | 프레임 해시·검출 결과 동일 |
| 카메라 앱 HTML·CSS·JS·상태·목록·사고 비활성 응답 6개 | 상태 코드·응답 본문 해시 동일 |
| 기존 CCTV 회귀 테스트 11개 | 통과 |
| 실제 CCTV tracking·ROI 분석·렌더링 | 저장 영상으로 3프레임 처리 성공 |
| 실제 로컬 대시보드·백엔드 HTTP | 총 15개 경로 응답 점검 통과 |
| 백엔드 영상 선택·제어 조회·잘못된 인덱스·탐지 수신 | 통과 |
| 두 경량화 구현의 단위 테스트 2개 | 통과 |
| 프론트엔드 `npm ci`, `npm run build` | 통과 |
| Python 전체 구문·학습 셸 스크립트 구문 | 통과 |

실제 모델 추론 24프레임은 CPU에서 수행했습니다. 검출 비교에는 모델 예측값을 그대로 사용했습니다. 실행 검증 도구의 보고서를 다시 생성해 추적·렌더링을 포함한 결과가 반복 실행에서도 일치함을 확인했습니다.

## 경량화 모듈

기존 `prune.py`는 `module_name` 인자를 지원하는 사용자 수정 구현이었습니다. 일반 PyTorch는 이 인자를 받지 않아 단순한 패키지 설치만으로 기존 경량화 코드를 실행할 수 없었습니다.

해당 구현을 바이트 변경 없이 `training/pruning.py`로 옮기고 두 `compress` 패키지에서 로컬로 불러오도록 연결했습니다. 설치된 PyTorch의 pruning 모듈은 수정하지 않습니다. 두 구현에서 각각 8·64·160채널 Conv의 50% 마스킹, forward 연산, BatchNorm pruning, 마스크 제거를 검증했습니다. 전체 데이터셋을 사용하는 재학습·정확도 평가는 수행하지 않았습니다.

## 재실행

공통 환경:

```bash
(cd main && python -m unittest discover -s tests -p 'test_*.py')
python tools/verify_runtime.py --report outputs/runtime.json
```

같은 입력·CPU 환경에서 후속 수정 결과 비교:

```bash
python tools/verify_runtime.py --compare outputs/runtime.json
```

학습 환경을 설치했다면:

```bash
python -m unittest discover -s training/tests -p 'test_*.py'
```

프론트엔드:

```bash
(cd main/frontend && npm ci && npm run build)
```

`outputs/` 보고서와 캐시·빌드 파일은 Git에서 제외합니다. 검증 도구는 원격 전송을 끄고 실제 시연 영상을 사용합니다. 백엔드의 ITS 조회만 고정된 응답으로 대체하며, 실제 Flask/CORS/WebSocket 확장은 등록한 상태로 HTTP 경로를 확인합니다.

## 검증 범위

USB 카메라 하드웨어 입력, 현재 ITS 실시간 스트림, Railway/Vercel의 운영 서버, 실제 네트워크의 WebSocket·WebRTC 연결은 이번 로컬 검증에 포함하지 않았습니다. 두 배포 폴더와 서버 설정은 유지했으며 웹 화면 소스도 변경하지 않았습니다. GPU별 추론 결과나 장시간 운용까지 동일하다고 보장하는 검사는 아닙니다.

환경 문서의 Python requirements는 기존 환경에서 의존성 해결을 확인했고, 프론트엔드는 lockfile 기준으로 설치·빌드했습니다. 완전히 비어 있는 새 OS나 Jetson 장치에서의 설치는 별도 확인이 필요합니다.
