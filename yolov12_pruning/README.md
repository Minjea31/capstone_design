# RVLAB_yolov12 - Lightweight Fire Detection Model

## 프로젝트 개요
본 프로젝트는 **YOLOv12 기반 화재 감지 모델의 경량화**를 목표로 하며, **엣지 디바이스 실시간 처리 성능 확보**를 위해 다음과 같은 기술 흐름을 따릅니다:

1. Channel Pruning (FPGM 기반)

본 문서는 그 중 **1단계: FPGM 기반 Channel Pruning**에 대한 내용을 다룹니다.

---

## 1. FPGM 기반 Channel Pruning

### 목적
- YOLOv12 모델에서 **불필요한 채널을 제거**하여 연산량과 파라미터 수를 줄이고, **엣지 환경에서의 실시간성 확보**.

### 사용 기술
- **FPGM (Filter Pruning via Geometric Median)** [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html)
- 각 convolution layer 내에서 **유사한 filter 간의 중심점(Geometric Median)을 기준으로 대표 filter만 유지**하고 나머지는 제거
- **글로벌 구조 수준의 pruning**이 가능

---

## 2. 모델 재구성 (Structure Rebuilding)

### 이유
- 일반적인 pruning은 weight를 0으로 만드는 soft pruning에 머물러, 실질적인 연산량 감소가 어려움
- 따라서, pruning 후 **불필요한 channel/filter를 아예 제거**하는 재구성 과정 필요:contentReference[oaicite:2]{index=2}

### 재구성 고려 요소
- **YOLOv12의 Residual/Concat/Split 구조**
- **C2f(C3k2), A2C2f 등 Hybrid 블록 구조**
- **Attention 연산 (Q/K index)** 일치 여부 보존

### 예시
- `Residual connection`에서 pruning 후 입출력 index가 맞지 않을 경우 → 0-weight channel 삽입하여 연산 흐름 유지
- `Concat 구조`에서 pruning 후 남은 채널 순서 정렬 필요 → offset 적용 후 정렬 및 불필요한 채널 제거

---

## 3. 실험 해본것들

### Fix-pruning-ratio

- FPGM

- k-means clustering

- cosine similarity

- qk_shared

- heuristic 한 cluster 갯수

- PCA로 2D 로 변환후 k-means clustering

- k-median

### layerwise-pruning

- GA 알고리즘

---

## 실험 모델 정보

| 항목 | 내용 |
|------|------|
| 베이스 모델 | YOLOv12 |
| 적용 방식 | Channel Filter Pruning (FPGM) |
| 대상 블록 | CSP 기반 C3k2, A2C2f, C2f 등 YOLO 주요 블록 |
| 목적 | 엣지 디바이스에서 화재 감지 실시간 처리 |

---

## 참고 자료

- He et al., *Filter Pruning via Geometric Median*, CVPR 2019

---

## 작성자
- 김민재 (지능로봇공학과, RVLAB)


