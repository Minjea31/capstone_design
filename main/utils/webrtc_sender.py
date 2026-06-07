"""YOLO PC annotated frame WebRTC sender."""

import asyncio
import json
import threading
import time
from fractions import Fraction
from urllib.parse import urlparse

import cv2
import numpy as np

from utils.app_config import (
    DETECTION_BACKEND_URL,
    DETECTION_WEBRTC_ENABLED,
    DETECTION_WEBRTC_FRAME_WIDTH,
    DETECTION_WEBRTC_FPS,
    DETECTION_WEBRTC_ICE_SERVERS,
    DETECTION_WEBRTC_SIGNALING_URL,
)


def _derive_webrtc_signaling_url(backend_url):
    if not backend_url:
        return ""
    parsed = urlparse(backend_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws/webrtc/sender"


def _load_webrtc_dependencies():
    try:
        import websockets
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
            VideoStreamTrack,
        )
        from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
        from av import VideoFrame
    except ImportError as exc:
        raise RuntimeError(
            "WebRTC 송출에는 aiortc, av, websockets 패키지가 필요합니다. "
            "pip install aiortc av websockets"
        ) from exc

    return {
        "websockets": websockets,
        "RTCConfiguration": RTCConfiguration,
        "RTCIceServer": RTCIceServer,
        "RTCPeerConnection": RTCPeerConnection,
        "RTCSessionDescription": RTCSessionDescription,
        "VideoStreamTrack": VideoStreamTrack,
        "candidate_from_sdp": candidate_from_sdp,
        "candidate_to_sdp": candidate_to_sdp,
        "VideoFrame": VideoFrame,
    }


def _parse_ice_servers(deps):
    try:
        raw_servers = json.loads(DETECTION_WEBRTC_ICE_SERVERS or "[]")
    except json.JSONDecodeError:
        raw_servers = []
    if not raw_servers:
        raw_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    ice_servers = []
    for server in raw_servers:
        if not isinstance(server, dict) or not server.get("urls"):
            continue
        ice_servers.append(
            deps["RTCIceServer"](
                urls=server.get("urls"),
                username=server.get("username"),
                credential=server.get("credential"),
            )
        )
    return ice_servers


class LatestFrameSource:
    def __init__(self, frame_width=0):
        self._lock = threading.Lock()
        self.frame_width = max(0, int(frame_width or 0))
        self._frame = self._build_placeholder()
        self._sequence = 0

    def _build_placeholder(self):
        frame = np.full((540, 960, 3), (18, 24, 34), dtype=np.uint8)
        cv2.putText(frame, "WebRTC waiting", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 236, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, "waiting for YOLO frame", (28, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 182, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (22, 22), (938, 518), (62, 78, 104), 2)
        return frame

    def update(self, frame):
        if frame is None:
            return
        output = frame
        height, width = output.shape[:2]
        if self.frame_width > 0 and width > self.frame_width:
            target_height = max(1, int(height * (self.frame_width / float(width))))
            output = cv2.resize(output, (self.frame_width, target_height), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._frame = output.copy()
            self._sequence += 1

    def get(self):
        with self._lock:
            return self._frame.copy(), self._sequence


def _build_frame_track_class(deps):
    class LatestFrameVideoTrack(deps["VideoStreamTrack"]):
        def __init__(self, frame_source, fps):
            super().__init__()
            self.frame_source = frame_source
            self.fps = max(1.0, float(fps))
            self.time_base = Fraction(1, 90000)
            self.frame_interval = 1.0 / self.fps
            self._start = None
            self._timestamp = 0

        async def recv(self):
            if self._start is None:
                self._start = time.time()
            else:
                wait = self._start + (self._timestamp / 90000) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)

            frame, _sequence = self.frame_source.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = deps["VideoFrame"].from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = self._timestamp
            video_frame.time_base = self.time_base
            self._timestamp += int(90000 * self.frame_interval)
            return video_frame

    return LatestFrameVideoTrack


class DetectionWebRTCClient:
    def __init__(self, signaling_url=None, enabled=None, fps=None):
        self.enabled = DETECTION_WEBRTC_ENABLED if enabled is None else bool(enabled)
        self.signaling_url = (
            signaling_url
            or DETECTION_WEBRTC_SIGNALING_URL
            or _derive_webrtc_signaling_url(DETECTION_BACKEND_URL)
        )
        self.fps = max(1.0, float(fps or DETECTION_WEBRTC_FPS))
        self.frame_source = LatestFrameSource(DETECTION_WEBRTC_FRAME_WIDTH)
        self.thread = None
        self.loop = None
        self.stop_event = threading.Event()
        self.deps = None

    def start(self):
        if not self.enabled:
            return
        if not self.signaling_url:
            print("WebRTC disabled: signaling URL is not configured")
            return
        try:
            self.deps = _load_webrtc_dependencies()
        except RuntimeError as exc:
            print(exc)
            return

        print(
            "WebRTC sender enabled:",
            f"signaling={self.signaling_url}",
            f"fps={self.fps:.1f}",
            f"width={self.frame_source.frame_width if self.frame_source.frame_width > 0 else 'original'}",
        )
        self.thread = threading.Thread(target=self._run_thread, daemon=True, name="webrtc-sender")
        self.thread.start()

    def submit_frame(self, frame):
        if self.enabled:
            self.frame_source.update(frame)

    def close(self):
        self.stop_event.set()
        if self.loop is not None:
            self.loop.call_soon_threadsafe(lambda: None)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_loop())
        finally:
            self.loop.close()

    async def _connect_loop(self):
        websockets = self.deps["websockets"]
        while not self.stop_event.is_set():
            peer_connections = {}
            try:
                print("WebRTC signaling connecting:", self.signaling_url)
                async with websockets.connect(
                    self.signaling_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2 * 1024 * 1024,
                ) as websocket:
                    print("WebRTC signaling connected")
                    await websocket.send(json.dumps({"type": "sender-ready"}))
                    heartbeat_task = asyncio.create_task(self._heartbeat(websocket))
                    try:
                        async for raw_message in websocket:
                            try:
                                await self._handle_message(websocket, peer_connections, json.loads(raw_message))
                            except Exception as exc:
                                print(f"WebRTC message handling failed: {type(exc).__name__}: {exc!r}")
                    finally:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"WebRTC signaling failed: {type(exc).__name__}: {exc!r}")
                    await asyncio.sleep(2)
            finally:
                await self._close_peer_connections(peer_connections)

    async def _heartbeat(self, websocket):
        while not self.stop_event.is_set():
            await asyncio.sleep(10)
            await websocket.send(json.dumps({"type": "ping"}))

    async def _handle_message(self, websocket, peer_connections, message):
        message_type = message.get("type")
        viewer_id = str(message.get("viewer_id") or "")
        if message_type in {"ping", "pong", "sender-ready"}:
            return
        if message_type == "viewer-left" and viewer_id:
            print(f"WebRTC viewer left: viewer={viewer_id}")
            await self._close_peer_connection(peer_connections, viewer_id)
            return
        if message_type == "viewer-joined" and viewer_id:
            print(f"WebRTC viewer joined: viewer={viewer_id}")
            return
        if message_type == "offer" and viewer_id:
            print(f"WebRTC offer received: viewer={viewer_id}")
            await self._handle_offer(websocket, peer_connections, viewer_id, message)
            return
        if message_type == "candidate" and viewer_id:
            await self._handle_candidate(peer_connections, viewer_id, message.get("candidate"))

    async def _handle_offer(self, websocket, peer_connections, viewer_id, message):
        await self._close_peer_connection(peer_connections, viewer_id)

        RTCConfiguration = self.deps["RTCConfiguration"]
        RTCPeerConnection = self.deps["RTCPeerConnection"]
        RTCSessionDescription = self.deps["RTCSessionDescription"]
        LatestFrameVideoTrack = _build_frame_track_class(self.deps)

        peer_connection = RTCPeerConnection(
            RTCConfiguration(_parse_ice_servers(self.deps))
        )
        peer_connections[viewer_id] = peer_connection
        peer_connection.addTrack(LatestFrameVideoTrack(self.frame_source, self.fps))

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"WebRTC peer state: viewer={viewer_id} state={peer_connection.connectionState}")
            if peer_connection.connectionState in {"failed", "closed", "disconnected"}:
                await self._close_peer_connection(peer_connections, viewer_id)

        @peer_connection.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            candidate_to_sdp = self.deps["candidate_to_sdp"]
            await websocket.send(
                json.dumps(
                    {
                        "type": "candidate",
                        "viewer_id": viewer_id,
                        "candidate": {
                            "candidate": "candidate:" + candidate_to_sdp(candidate),
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        offer = RTCSessionDescription(sdp=message.get("sdp", ""), type="offer")
        await peer_connection.setRemoteDescription(offer)
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
        await websocket.send(
            json.dumps(
                {
                    "type": "answer",
                    "viewer_id": viewer_id,
                    "sdp": peer_connection.localDescription.sdp,
                },
                ensure_ascii=False,
            )
        )
        print(f"WebRTC answer sent: viewer={viewer_id}")

    async def _handle_candidate(self, peer_connections, viewer_id, candidate_data):
        peer_connection = peer_connections.get(viewer_id)
        if peer_connection is None or not candidate_data:
            return

        candidate_text = str(candidate_data.get("candidate", ""))
        if candidate_text.startswith("candidate:"):
            candidate_text = candidate_text[len("candidate:") :]
        if len(candidate_text.split()) < 8:
            return
        candidate = self.deps["candidate_from_sdp"](candidate_text)
        candidate.sdpMid = candidate_data.get("sdpMid")
        candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
        await peer_connection.addIceCandidate(candidate)

    async def _close_peer_connection(self, peer_connections, viewer_id):
        peer_connection = peer_connections.pop(str(viewer_id), None)
        if peer_connection is not None:
            await peer_connection.close()

    async def _close_peer_connections(self, peer_connections):
        for viewer_id in list(peer_connections):
            await self._close_peer_connection(peer_connections, viewer_id)
