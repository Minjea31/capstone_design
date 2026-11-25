# CCTV 대시보드 프론트엔드

Next.js App Router 기반 화면입니다. 백엔드의 CCTV·저장 영상 목록, 차량 수와 추세, 혼잡·사고 상태를 표시하고 입력 선택을 제어합니다. WebSocket 상태 수신, WebRTC 영상 표시와 JPEG 대체 경로를 지원합니다.

## 설치·실행

검증에 사용한 Node.js는 22입니다.

```bash
# 저장소 루트에서
cd main/frontend
npm ci
NEXT_PUBLIC_API_URL=http://localhost:5000 npm run dev
```

브라우저에서 `http://localhost:3000`에 접속합니다. API 주소 끝에는 `/`를 붙이지 않습니다. 백엔드 포트가 5001이라면 환경변수도 맞춥니다. `.env.local`에 저장해도 되며 해당 파일은 Git에서 제외합니다.

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

빌드와 실행:

```bash
npm run build
npm start
```

## Vercel

프로젝트 루트는 `main/frontend`입니다. `NEXT_PUBLIC_API_URL`을 Railway 백엔드 주소로 설정합니다. 브라우저용 환경변수는 빌드에 반영되므로 변경한 뒤 다시 빌드·배포합니다.

WebRTC ICE 서버 설정은 `NEXT_PUBLIC_WEBRTC_ICE_SERVERS`를 사용합니다. 분석 PC의 설정 예시는 [송신기 안내](../README.md#webrtc-영상-송출)를 참고합니다.

공통 환경 구축은 [설치 문서](../../docs/setup.md)에 있습니다. 이번 폴더 정리에서도 기존 Vercel 배포 루트와 화면 코드는 유지했습니다.
