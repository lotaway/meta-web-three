# meta-note元笔记

这是一个基于 Electron + React + TypeScript + Vite 构建的桌面应用。

## Environment Configuration

The application uses environment variables for configuration.

### 1. Setup Environment File

```bash
cp .env.example .env
```

### 2. Configure Voice API

Edit `.env` and set the voice service endpoint:

```bash
VITE_VOICE_API_URL=http://localhost:8000
```

## Local HTTP API authentication

The Electron process exposes its HTTP API on `127.0.0.1:${WEB_SERVER_PORT}` (default `5051`). Every route requires a session token:

- The renderer obtains it through the `getLocalApiToken` preload method and sends `Authorization: Bearer <token>`.
- SSE endpoints that cannot set headers (`/api/tts/download`) accept `?token=<token>`.
- External callers can pin a known value with `LOCAL_API_TOKEN` in `.env`; otherwise a random token is generated on each start.
- Only loopback origins (`file://`, the Vite dev server and the API itself) are allowed; wildcard CORS is disabled.

## LLM models and monitor windows

`POST /v1/chat/completions` dispatches on `payload.model`:

| model | behavior |
|---|---|
| `local` (default) | `LLMService` or the `LOCAL_LLM_PROVIDER` SSE proxy |
| `chatgpt` | in-app ChatGPT monitor window, SSE chunks converted to OpenAI deltas |
| `deepseek` | in-app DeepSeek monitor window, SSE chunks converted to OpenAI deltas |

Monitor windows keep `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` and use a dedicated `monitor-preload` that exposes only named channels. Session tokens for both providers are written through the token-protected `POST /v1/auth/token` route.

## Running the Application

使用yarn而不是npm可以避免很多间接依赖版本冲突问题。

### Development mode
```bash
yarn dev
```

### Build application
```bash
yarn build
```

## Features

### Voice Transcription & Subtitles

Real-time transcription of microphone or system audio with a customizable overlay.

1. **Select Source**
   - **Microphone**: Direct environment recording.
   - **System Audio**: Select a specific window or tab from the list.

2. **Start Transcription**
   - Click **"Start Recording"** in the control panel.
   - Data is streamed to the backend for processing.

3. **Floating Subtitles**
   - Click **"Open Subtitles"** to show the overlay.
   - Click ⚙️ on the overlay to customize:
     - Font size and color
     - Text outline color and width

4. **File Transcription**
   - Upload local audio/video files (MP3/MP4) for offline transcription.

### Backend Requirements

Requires a compatible backend service:
- `POST /voice/to/text` (Supports SSE stream and chunked uploads)
