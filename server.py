# SPDX-License-Identifier: Apache-2.0
"""Twilio Media Streams phone voice bot server.

- POST /twiml   TwiML for incoming calls: <Connect><Stream> points at /media here
- WS   /media   Twilio Media Streams bidirectional WebSocket
"""

import asyncio
import base64
import json
import logging
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import config
from asr_engine import ASRSession
from audio_codec import ulaw8k_to_pcm16_16k
from call_session import CallSession, STATE_LISTENING, STATE_SPEAKING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("server")

app = FastAPI(title="qwen3-tts-twilio-bot")

# streamSid -> CallSession
_sessions: dict[str, CallSession] = {}


def _public_ws_url(request: Request) -> str:
    """wss URL of this service's /media (behind ngrok: use forwarded headers)."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    proto = request.headers.get("x-forwarded-proto", "https")
    ws_proto = "wss" if proto in ("https", "wss") else "ws"
    return f"{ws_proto}://{host}/media"


@app.get("/health")
async def health():
    return {"ok": True, "sessions": len(_sessions)}


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    # GET allowed only as a sanity check; Twilio always POSTs.
    # (A GET here is usually a browser misconfig: Twilio would treat the
    # 405 body as invalid TwiML and the call fails / WebSocket gets 403.)
    ws_url = _public_ws_url(request)
    logger.info("incoming call, stream url=%s", ws_url)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>"""
    return Response(content=xml, media_type="application/xml")


async def _endpointing_loop(session: CallSession):
    """Poll for user turn end: in listening state, once VAD silence exceeds the
    threshold and speech was accumulated -> finalize ASR and start a new turn."""
    last_voice_at = time.monotonic()
    prev_state = STATE_LISTENING
    prev_voice_ms = 0.0
    while not session._closed:
        await asyncio.sleep(0.2)
        state = session.state
        voice_ms = session.vad._voice_ms
        # Voice arrived since the last poll (works in every state, including
        # right after a barge-in flipped the state to LISTENING):
        if voice_ms > prev_voice_ms:
            last_voice_at = time.monotonic()
        prev_voice_ms = voice_ms
        if state != STATE_LISTENING:
            prev_state = state
            continue
        prev_state = state
        if voice_ms > 0:
            continue
        silence = time.monotonic() - last_voice_at
        if silence > config.TURN_END_SILENCE_SEC + 0.5:
            logger.info("[%s] endpoint: silence=%.1fs, finalizing ASR",
                        session.stream_sid, silence)
            final = await session.asr.finalize()
            if final.strip():
                session._interrupted = False
                task = asyncio.create_task(session.handle_turn(final))
                session._turn_tasks.add(task)
                task.add_done_callback(session._turn_tasks.discard)
                await asyncio.sleep(0)
            else:
                logger.info("[%s] endpoint: ASR empty, no turn", session.stream_sid)
            last_voice_at = time.monotonic()


@app.websocket("/twiml")
async def twiml_ws_fallback(ws: WebSocket):
    """Safety net: some Twilio numbers are configured to open the Media Stream
    on the webhook URL itself (GET /twiml), not /media. Accept here and log so
    the call still connects instead of getting a 403."""
    logger.warning("Media Stream opened on /twiml (fallback); webhook should point <Stream> at /media")
    await media_stream(ws)


@app.websocket("/media")
async def media_stream(ws: WebSocket):
    await ws.accept()
    session: CallSession | None = None
    stream_sid: str | None = None
    send_lock = asyncio.Lock()

    async def send_json(msg: dict):
        async with send_lock:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                pass  # call already ended, ignore send failures

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.info("twilio websocket connected")

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                session = CallSession(ws, send_json,
                                     custom_params=msg["start"].get("customParameters", {}))
                session.stream_sid = stream_sid
                session.asr = ASRSession(sid=stream_sid[:12])
                _sessions[stream_sid] = session
                asyncio.create_task(_endpointing_loop(session))
                await session.start()

            elif event == "media":
                if session:
                    ulaw = base64.b64decode(msg["media"]["payload"])
                    try:
                        await session.on_inbound_media(ulaw)
                    except (asyncio.CancelledError, WebSocketDisconnect):
                        raise
                    except Exception:
                        logger.exception("inbound media handling failed")

            elif event == "mark":
                if session:
                    await session.on_mark(msg["mark"]["name"])

            elif event == "dtmf":
                if session:
                    await session.on_dtmf(msg["dtmf"]["digit"])

            elif event == "stop":
                logger.info("stream stopped: %s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("twilio websocket disconnected: %s", stream_sid)
    except Exception:
        logger.exception("media stream error: %s", stream_sid)
    finally:
        if session:
            await session.close()
        if stream_sid:
            _sessions.pop(stream_sid, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="info",
        timeout_keep_alive=30,
    )
