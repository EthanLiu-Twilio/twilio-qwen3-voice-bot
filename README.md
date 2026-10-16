# Qwen3-TTS + Twilio Media Streams Phone Voice Bot

Twilio inbound call → bidirectional Media Streams WebSocket → local server:
hearing, thinking and speaking run **entirely through the local oMLX OpenAI-compatible HTTP API** (`/v1/audio/transcriptions`, `/v1/chat/completions`, `/v1/audio/speech`). Supports barge-in interruption. Trilingual: Japanese / Mandarin Chinese / English.

## Architecture

```
phone ──call── Twilio ──wss── ngrok ──> server.py (:8765)
                                          │
        inbound 8kHz μ-law ──> decode + upsample 16k ──> oMLX /audio/transcriptions
                                          │  VAD silence -> end-of-turn
                                          v
                                   oMLX /chat/completions (streaming LLM, per sentence)
                                          │
        outbound 8kHz μ-law <── encode <── oMLX /audio/speech (stream wav 24k)
                                          │
        user interrupts 0.4s ──> send clear to stop playback (barge-in)
```

- The Twilio side is fixed **8kHz μ-law mono**, sent on a 20ms cadence (160 bytes/block).
- This process loads no models; it only calls oMLX HTTP endpoints. Models (load/unload/scheduling) are managed by oMLX.
- One concurrent call at a time.

## Prerequisites

1. **oMLX** running on `localhost:8000` with a working API key and these models (visible via `GET /v1/models`):
   - `Qwen3-TTS-12Hz-1.7B-Base-fp32` (TTS; the Base model needs ref_audio for voice cloning)
   - `Qwen3-ASR-1.7B-bf16` (ASR)
   - Any chat LLM (default `Qwen3.8-Flash-Next-oQ4e-mtp`)
2. **Reference audio**: record 3–10 seconds of the voice you want (quiet room, single speaker), transcribe it verbatim, put the paths in `.env`. For trilingual use, cut one ref per language (`REF_AUDIO_PATH_JA/ZH/EN`) to avoid cross-language accent contamination.
3. **Twilio**: a phone number; its Voice webhook points at this service.
4. **ngrok**: `brew install ngrok` (or download from the website).
5. Python dependencies (lightweight, no mlx):
   ```bash
   pip install fastapi uvicorn httpx numpy
   ```

## Start

```bash
cd ~/Documents/qwen3tts-test-twilio
cp .env.example .env   # fill in oMLX / REF_AUDIO / tuning values
source .env

# 1. start the server
python server.py

# 2. in another terminal, expose it publicly
ngrok http 8765
```

Put the ngrok `https://xxxx.ngrok-free.dev` URL into the Twilio console:
Phone Number → Voice Configuration → **A Call Comes In** → `https://xxxx.ngrok-free.dev/twiml`.
(Free-tier URLs end in `.ngrok-free.dev`; check ngrok's local API at
`http://127.0.0.1:4040/api/tunnels` for the current one.)

> When ngrok restarts with a new domain, update the Twilio webhook accordingly.

## Test Call

Dial your Twilio number from a phone:
1. Within ~3s of pickup you hear the welcome line (first chunk includes TTS cold start).
2. Say a sentence; after 0.8s of silence the bot responds.
3. Interrupt for 0.4s while the bot speaks: it stops playback immediately and listens to you.

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI: `POST /twiml` + `WS /media` (Twilio protocol) |
| `call_session.py` | Call state machine: listening/thinking/speaking + barge-in |
| `tts_engine.py` | oMLX `/audio/speech` streaming synthesis (24k wav → 8k μ-law), per-language voice-cloning refs |
| `asr_engine.py` | oMLX `/audio/transcriptions` wrapper (buffer + finalize on endpoint) |
| `llm_client.py` | oMLX `/chat/completions` streaming chat, split by sentence |
| `audio_codec.py` | G.711 μ-law codec (verified against ffmpeg) + numpy 8k→16k upsampling |
| `config.py` | All configuration |

## Known Limitations & Follow-ups

- **First chunk ~3s**: Base TTS cold start (model load/generation on the oMLX side). Improvement directions: keep the model resident in oMLX, switch to the CustomVoice model (no ref encoding), or use a cloud TTS.
- **ASR is not truly streaming**: the whole segment is transcribed once at endpointing. If oMLX supports SSE streaming transcription (`stream=true`) this can be optimized further.
- **Single concurrent call**: models are singletons; multiple calls queue serially.
- **Interrupted speech not in history**: half-sentence ASR text at barge-in is not stored in the dialog history.
- The ngrok free tier domain changes on every restart.

## Barge-in Mechanism

While the bot is `SPEAKING`, sustained user speech ≥400ms (tunable via `BARGE_IN_MS`) triggers an interruption:
1. Send a `clear` event to Twilio to flush the playback buffer
2. Set `_interrupted=True`; the `speak()` loop notices and exits promptly
3. Reset the VAD; state returns to `LISTENING`

**Real-device test**: interrupt the bot for 0.4s while it speaks — it stops immediately and listens to you.
