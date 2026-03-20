"""Flask 기반 CCTV 웹 UI.

표시는 브라우저가 원본 HLS를 직접 재생하고,
서버는 CCTV 목록과 선택된 스트림 URL만 제공합니다.
"""

import sys

try:
    from flask import Flask, jsonify, render_template
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Flask가 현재 실행 중인 파이썬 환경에 설치되어 있지 않습니다.\n"
        f"현재 인터프리터: {sys.executable}\n"
        "IDE 인터프리터를 /home/user/anaconda3/bin/python 으로 바꾸거나,\n"
        "해당 인터프리터에 Flask를 설치한 뒤 다시 실행하세요."
    ) from exc

from werkzeug.serving import WSGIRequestHandler

from yolo_opencv import fetch_cctv_dataframe, resolve_stream_url

app = Flask(__name__)


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.path.startswith("/api/status"):
            return
        super().log_request(code, size)


class VideoService:
    def __init__(self):
        self.cctv_df = fetch_cctv_dataframe()
        self.cctv_names = self.cctv_df["cctvname"].astype(str).fillna("").tolist()
        self.selected_index = 0
        self.selected_name = self.cctv_names[0] if self.cctv_names else "-"
        self.selected_stream_url = None
        self.traffic_count = 0
        self.accident_status = "-"
        self.stream_status = "준비 중"
        self._select_first_available_stream()

    def _resolve_stream_for_index(self, index):
        row = self.cctv_df.iloc[index]
        name = str(row["cctvname"])
        raw_url = row["cctvurl"]
        stream_url = resolve_stream_url(raw_url)
        if not stream_url:
            raise RuntimeError(f"스트림 URL을 해석하지 못했습니다: {name}")
        return name, stream_url

    def _select_first_available_stream(self):
        last_error = None
        for index in range(len(self.cctv_names)):
            try:
                self.select_index(index)
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"열 수 있는 CCTV 스트림이 없습니다: {last_error}")

    def select_index(self, index):
        if index < 0 or index >= len(self.cctv_names):
            raise IndexError("CCTV 인덱스 범위를 벗어났습니다.")

        name, stream_url = self._resolve_stream_for_index(index)
        self.selected_index = index
        self.selected_name = name
        self.selected_stream_url = stream_url
        self.accident_status = "-"
        self.stream_status = "연결됨"

    def get_status(self):
        return {
            "selected_index": self.selected_index,
            "selected_name": self.selected_name,
            "traffic_count": self.traffic_count,
            "accident_status": self.accident_status,
            "stream_status": self.stream_status,
            "cctv_count": len(self.cctv_names),
        }

    def get_cctv_items(self):
        return [
            {
                "index": index,
                "name": name,
                "selected": index == self.selected_index,
            }
            for index, name in enumerate(self.cctv_names)
        ]

    def get_selected_stream_payload(self):
        return {
            "index": self.selected_index,
            "name": self.selected_name,
            "stream_url": self.selected_stream_url,
        }


service = VideoService()


@app.route("/")
def index():
    return render_template(
        "index.html",
        cctvs=service.get_cctv_items(),
        status=service.get_status(),
        selected_stream=service.get_selected_stream_payload(),
    )


@app.route("/api/status")
def api_status():
    return jsonify(service.get_status())


@app.route("/api/cctvs")
def api_cctvs():
    return jsonify({"items": service.get_cctv_items()})


@app.route("/api/stream")
def api_stream():
    return jsonify(service.get_selected_stream_payload())


@app.route("/api/stream/<int:index>")
def api_stream_by_index(index):
    try:
        name, stream_url = service._resolve_stream_for_index(index)
    except (RuntimeError, IndexError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            "index": index,
            "name": name,
            "stream_url": stream_url,
        }
    )


@app.route("/api/select/<int:index>", methods=["POST"])
def api_select(index):
    try:
        service.select_index(index)
    except (RuntimeError, IndexError) as exc:
        service.stream_status = "연결 실패"
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "status": service.get_status(), "stream": service.get_selected_stream_payload()})


def main():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, request_handler=QuietRequestHandler)


if __name__ == "__main__":
    main()
