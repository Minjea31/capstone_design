<<<<<<< HEAD
# Capstone Design CCTV Detection Pipeline

## 프로젝트 소개

본 프로젝트는 **고속도로 CCTV 영상을 실시간으로 수신**하고,  
학습 및 경량화한 **Pruned YOLO 모델로 차량 객체를 탐지**한 뒤,  
탐지 결과를 **웹 UI에서 시각화**하는 Capstone Design 프로젝트입니다.

즉, 전체 목표는 아래와 같습니다.

`CCTV 영상 수집 -> Pruned 모델 추론 -> 탐지 결과 생성 -> 웹 화면 표시`

## 팀 구성

- 김민재
- 이시온
- 지연수

## 실행 방법

현재 프로젝트는 아래 명령으로 실행합니다.

```bash
python main/main.py --use-roi
```

## 핵심 목표

- CCTV 실시간 스트림을 안정적으로 수신한다.
- Pruned YOLO 기반으로 차량을 탐지한다.
- 탐지 결과를 웹에서 실시간으로 확인할 수 있도록 제공한다.
- 필요 시 ROI(관심영역)를 설정해 특정 구간만 분석할 수 있도록 확장한다.

## 전체 파이프라인

### 1. CCTV 영상 수신

- ITS Open API에서 CCTV 목록과 스트림 정보를 받아온다.
- 선택한 CCTV의 `cctvurl`을 기반으로 실제 재생 가능한 스트림 주소를 찾는다.
- OpenCV `VideoCapture`로 영상을 열고 프레임을 지속적으로 읽는다.

### 2. Pruned 모델 기반 객체 탐지

- `pruned.pt`와 `pruned.yaml`을 이용해 경량화된 YOLO 모델을 로드한다.
- 입력 프레임마다 차량 클래스 기준으로 detection/tracking을 수행한다.
- 탐지된 객체의 바운딩 박스와 confidence를 프레임 위에 그린다.

### 3. 분석 결과 생성

- 탐지된 박스 개수를 기반으로 현재 교통량을 계산한다.
- 필요하면 ROI를 이용해 특정 차선 또는 특정 구역 중심의 분석으로 확장할 수 있다.
- 후속 단계에서는 사고 위험도, 상행/하행 분리, 구간별 통계 등으로 연결할 수 있다.

### 4. 웹 UI 실시간 시각화

- Flask 서버에서 CCTV 상태와 분석 결과를 관리한다.
- 웹 페이지에서 선택한 CCTV 영상과 탐지 결과를 표시한다.
- 교통량, 상태 정보, 추세 그래프 등을 대시보드 형태로 보여준다.

## 파이프라인 요약

1. CCTV 목록 조회
2. 스트림 URL 연결
3. OpenCV로 프레임 수신
4. Pruned YOLO 모델 추론
5. 바운딩 박스가 그려진 결과 프레임 생성
6. Flask 서버로 상태 전달
7. 웹 UI에서 실시간 표시

## 파일별 역할 정리

### `main.py`

전체 시스템을 실행하는 진입점입니다.

- 파이프라인 실행 옵션을 받음
- `PrunedCCTVPipeline` 생성
- Flask UI 서버 실행
- 최종적으로 `CCTV 수신 -> 추론 -> 웹 표시` 흐름을 시작

정리하면, **프로젝트 실행 시작 파일**입니다.

### `pruned_yolo_cctv.py`

Pruned YOLO 모델을 사용해 CCTV 영상에서 객체 탐지를 수행하는 메인 추론 스크립트입니다.

- CCTV 목록을 조회하고 특정 CCTV를 선택
- 스트림 URL을 해석하고 OpenCV로 연결
- `pruned.pt`, `pruned.yaml` 기반 모델 로드
- 프레임 단위 detection/tracking 수행
- 필요 시 ROI 내부 객체만 필터링
- 결과 바운딩 박스를 프레임에 시각화
- 단독 실행 시 OpenCV 창으로 실시간 결과 확인 가능
- 클래스 기반 파이프라인(`PrunedCCTVPipeline`) 제공

정리하면, **모델 추론 중심 파일**입니다.

### `roi.py`

영상 위에 ROI(Region of Interest, 관심 영역)를 직접 지정하고 저장하는 도구입니다.

- 마우스로 다각형 ROI를 그림
- 라벨별로 ROI를 구분 가능
- ROI 정보를 JSON 파일로 저장/불러오기 가능
- 이후 특정 구간만 분석하거나 차선 단위 분석으로 확장할 때 활용 가능

정리하면, **분석 구역 설정 파일**입니다.

### `cctv_ui.py`

탐지 결과를 웹에서 확인하기 위한 Flask 기반 UI 서버입니다.

- 추론 파이프라인 객체를 받아 웹 서비스로 연결
- 백그라운드에서 처리된 프레임을 지속적으로 읽음
- MJPEG 방식으로 탐지 결과 프레임을 웹에 전달
- 현재 선택된 CCTV, 교통량, 상태 정보 등을 API로 제공
- `templates/index.html`, `static/style.css`와 함께 대시보드 UI 구성

정리하면, **웹 시각화 및 서비스 제공 파일**입니다.

## 현재 코드 기준 구성 흐름

현재 프로젝트는 아래 구조로 이해하면 됩니다.

- `main.py`: 전체 파이프라인 실행
- `pruned_yolo_cctv.py`: Pruned 모델 기반 탐지 테스트/실행
- `roi.py`: ROI 설정 및 저장
- `cctv_ui.py`: 웹 기반 모니터링 화면
- `templates/index.html`: CCTV 대시보드 화면 템플릿
- `static/style.css`: UI 스타일

## 권장 실행 흐름

캡스톤 발표나 시연 기준으로는 아래 순서가 가장 자연스럽습니다.

1. 필요한 경우 `roi.py`로 관심영역을 먼저 설정한다.
2. `pruned_yolo_cctv.py`로 pruned 모델이 정상적으로 탐지되는지 확인한다.
3. 최종 시연은 `main.py`를 실행해 전체 파이프라인을 웹으로 띄운다.

## 실행 예시

기본 실행:

```bash
python main.py
```

ROI 적용 실행:

```bash
python main.py --use-roi --roi-path roi/roi_data_2.json
```

## 향후 확장 방향

- ROI 내부 객체만 집계하도록 로직 확장
- 교통량 기반 혼잡도 분석
- 사고 위험도 예측 로직 고도화
- 탐지 결과 저장 및 로그 누적 기능 추가

## 한 줄 정리

이 프로젝트는 **실시간 CCTV 영상에 대해 Pruned YOLO로 차량을 탐지하고, 그 결과를 웹 대시보드로 보여주는 교통 모니터링 시스템**입니다.
=======
DETECTION_BACKEND_URL=https://detection-demo-production.up.railway.app DETECTION_LOG_SEND=1 python3 pruned_yolo_cctv.py
>>>>>>> 766f10542d455e1fb2d4f71a0aac74200399ebbb
