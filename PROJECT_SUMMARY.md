# Project Summary: Trilingual Real-time Voice Phone Bot on Twilio

> A real phone number ↔ AI voice call bot doing all inference locally on an
> Apple Silicon Mac: real-time Japanese / Mandarin Chinese / English conversation,
> cloned human voice, interruptible.
> Updated: 2026-10-16

## 1. Architecture in One Line

```
phone dials → Twilio Media Streams → ngrok tunnel → local FastAPI (:8765)
                μ-law 8kHz WS                │
                                            ├─ ASR → oMLX Qwen3-ASR-1.7B      (:8000)
                                            ├─ LLM → oMLX Qwen3.8-Flash-Next   (:8000)
                                            └─ TTS → oMLX Qwen3-TTS-1.7B-Base  (:8000)
                                                    (voice-clone ref = real recording "Ellin")
```

- **All AI inference runs locally on the Mac** (oMLX / MLX stack); Twilio only
  provides phone ingress and audio forwarding.
- **Hardware:** developed on a 128GB M5 Max with all three models resident
  (~85–90GB measured).
  Smaller-memory Macs work too, but must swap the chat LLM (default
  `Qwen3.8-Flash-Next-oQ4e-mtp`) for a smaller MLX model via `LLM_MODEL`
  (e.g. an 8B–14B Qwen3-class model); TTS + ASR together need ~10–15GB.
- Public ingress via ngrok free domain: `https://<your-random>.ngrok-free.dev`
  (free-tier URLs end in `.ngrok-free.dev`; the domain changes on every restart —
  check your ngrok terminal or inspector for the current URL)
- No cloud API fees, no data leaves the machine (except the Twilio phone channel itself).

## 2. Code Modules

| File | Responsibility |
|---|---|
| [server.py](server.py) | FastAPI: `POST /twiml` serves TwiML, `WS /media` carries the Twilio Media Streams audio, `/health` health check; event dispatch (connected/start/media/mark/dtmf/stop) |
| [call_session.py](call_session.py) | Per-call state machine: listening → thinking → speaking; energy VAD, barge-in, turn scheduling |
| [asr_engine.py](asr_engine.py) | Streaming ASR (oMLX `/v1/audio/transcriptions`), endpointing by silence, domain-vocabulary prompt biasing + prompt-leak filtering |
| [llm_client.py](llm_client.py) | Streaming chat (oMLX `/v1/chat/completions`), splits by punctuation and sends sentence-by-sentence to TTS |
| [tts_engine.py](tts_engine.py) | Voice-cloning TTS (oMLX `/audio/speech`), language detection auto-switches ref audio, ffmpeg streaming 24k→8k resampling + μ-law encoding |
| [audio_codec.py](audio_codec.py) | μ-law ↔ PCM codecs; pure-numpy 2:1 upsampling (8k→16k), no subprocess |
| [config.py](config.py) | Central config + minimal .env loader |
| [ref_audios/](ref_audios/) | Real human reference audio (Ellin trilingual recording + per-language cuts) |

## 3. Capabilities

### 3.1 Trilingual real-time conversation (ja / zh / en)
- **ASR**: `language=auto` auto-detection, mixed-language OK; `ASR_PROMPT` injects
  domain vocabulary (中国、中学生、オーストラリア、メルボルン、水道局、蛇口…) to
  improve recognition of ambiguous phonemes.
- **LLM**: system prompt enforces "reply in the same language the user last used",
  never switches on its own.
- **TTS**: `detect_lang()` decides text language by kana/hanzi/ASCII character
  counts; **each language uses its own native-speaker reference audio** to remove
  cross-language accent contamination (a trilingual ref previously gave Chinese
  output a Japanese accent).

### 3.2 Cloned human voice (fixed male voice)
- Reference audio: `Ellin St 5.m4a` (33s trilingual real male recording, F0≈103Hz,
  self-measured to confirm male).
- Per-language cuts: `ref_ja.wav` (10.5s), `ref_zh.wav` (9.5s); English not yet
  cut — falls back to the full trilingual ref.
- Qwen3-TTS Base only supports ref-based cloning (`/v1/audio/voices` offers no
  named voices for this model).

### 3.3 Realtime feel & interruption
- **20ms send cadence**: 160 bytes μ-law per tick, paced against an absolute
  clock so the buffer never overflows.
- **Fully streaming**: LLM emits sentence-by-sentence → TTS streams while
  synthesizing → low first-audio latency.
- **Barge-in**: user sustains speech ≥400ms while the bot speaks → send `clear`
  to stop playback; purely cooperative-flag exit (see §4, blood-and-tears lessons).
- **TTS speed**: `TTS_SPEED=1.15` (oMLX native speed param, tunable 1.0–1.3).

## 4. Traps We Hit (read before changing code)

1. **Never `task.cancel()` inside the WebSocket path.**
   The cancel propagates a CancelledError through `ws.receive_text()` and kills
   the entire call connection (looks like "the call hung up by itself").
   Correct approach: `_interrupted` cooperative flag + a `handle_turn` wrapper
   that swallows exceptions; cancelling inside `close()` is safe (WS already down).
2. **The ASR prompt can get "spoken back".**
   Instructional prompts ("please transcribe the spoken conversation…") come back
   transcribed as user speech. Keep prompts as bare noun phrases and filter leaks
   by substring in finalize.
3. **TTS reference-audio language contaminates the output accent.**
   A trilingual ref (Japanese first) → Chinese output with a Japanese accent.
   Use per-language refs.
4. **macOS `say` trap**: Rocko/Reed/Eddy/Grandpa only emit 0.016s silent stubs;
   Kyoko works but is female (F0≈255Hz). Resolved by using real recordings.
5. **Apostrophes in .env**: an unquoted `I'm` makes zsh `source .env` fail to
   parse (exit 126). Wrap values in single quotes, or rewrite apostrophes out.
6. **Base64 ref too big for curl**: a 33s ref's base64 exceeds curl's command
   line limit; send the JSON body via python urllib/requests instead.

## 5. How to Run

```bash
cd ~/Documents/qwen3tts-test-twilio
# 1. oMLX must already be running on :8000 (TTS/ASR/LLM models loaded)
# 2. start the bot
source .env && python server.py        # :8765
# 3. point ngrok at :8765; Twilio number Voice webhook -> ngrok domain /twiml
curl http://localhost:8765/health      # {"ok":true,"sessions":0}
```

## 6. Tuning Quick Reference (.env)

| Variable | Current | Meaning |
|---|---|---|
| `TTS_SPEED` | 1.15 | speed multiplier, 1.0–1.3 |
| `REF_AUDIO_PATH_JA/ZH` | ref_ja/zh.wav | per-language cloning refs |
| `BARGE_IN_MS` | 400 | sustained-speech threshold for interruption |
| `TURN_END_SILENCE_SEC` | 0.8 | silence duration that means the user is done |
| `ASR_LANGUAGE` | auto | can pin ja/zh/en |
| `ASR_PROMPT` | trilingual domain vocab | recognition biasing words |

## 7. TODO / Watch Items

- [ ] Live-call validation: 1.15x speed feel, per-language ref accent removal, cooperative barge-in stability
- [ ] Twilio webhook must point at the current ngrok domain with method POST (`https://<your-ngrok-domain>/twiml`)
- [ ] Add an English ref: cut ~10s of English from ellin_st5.wav → `ref_audios/ref_en.wav` + set `REF_AUDIO_PATH_EN/REF_TEXT_EN`
- [ ] The 33s fallback ref adds ~0.3–0.5s to the first synthesis (base64 is cached, so impact is limited)
- [ ] ngrok free-tier domain changes on restart; consider a fixed domain or Cloudflare Tunnel
