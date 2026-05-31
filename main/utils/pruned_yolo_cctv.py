from utils.detector import load_model, run_inference


class PrunedYOLODetector:
    """Pruned YOLO detection 전용 래퍼."""

    def __init__(self):
        self.model = load_model()

    def predict(self, frame):
        return run_inference(self.model, frame)


def main():
    raise SystemExit("`pruned_yolo_cctv.py`는 detector 전용입니다. 전체 실행은 `python3 main.py --use-roi`를 사용해주세요.")


if __name__ == "__main__":
    main()
