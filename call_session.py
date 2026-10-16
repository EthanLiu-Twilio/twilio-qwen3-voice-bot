# SPDX-License-Identifier: Apache-2.0
"""Per-call state machine: listening -> thinking -> speaking -> listening …

Responsibilities:
- Receive Twilio inbound μ-law, decode + VAD + feed ASR
- Silence detection marks end of user speech -> streaming LLM -> per-sentence
  TTS -> send to Twilio on a 20ms cadence
- Barge-in: sustained user speech while speaking -> send clear to stop playback
"""

import asyncio
import base64
import logging
import time

import numpy as np

import config
from asr_engine import ASRSession
from audio_codec import ulaw8k_to_pcm16_16k, ulaw_to_pcm16
from llm_client import stream_chat
from tts_engine import synthesize_stream

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"


class EnergyVAD:
    """Simple energy VAD: frames whose RMS exceeds the threshold count as speech."""

    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        self._voice_ms = 0.0
        self._silence_ms = 0.0

    def feed(self, pcm16k: bytes) -> tuple[bool, float]:
        """Returns (is_speech, voiced_duration_ms)."""
        pcm = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return self._voice_ms > 0, self._voice_ms
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        frame_ms = len(pcm) / config.ASR_SAMPLE_RATE * 1000.0
        if rms > self.threshold:
            self._voice_ms += frame_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms
            if self._silence_ms > 300:  # brief pauses do not reset the speech accumulator
                self._voice_ms = 0.0
        return self._voice_ms > 0, self._voice_ms

    def reset(self):
        self._voice_ms = 0.0
        self._silence_ms = 0.0


class CallSession:
    """One call. server.py holds this object for the lifetime of the Twilio WS."""

    def __init__(self, ws, send_json, *, custom_params: dict | None = None):
        self.ws = ws
        self._send_json = send_json
        self.custom_params = custom_params or {}
        self.stream_sid: str | None = None
        self.state = STATE_IDLE
        self.history: list[dict] = []
        self.asr = ASRSession(sid="")
        self.vad = EnergyVAD()
        self._interrupted = False
        self._turn_tasks: set[asyncio.Task] = set()
        self._last_activity = time.monotonic()
        self._closed = False

    # -- Twilio send ------------------------------------------------------

    async def _send_media(self, ulaw: bytes):
        msg = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"track": "outbound", "payload": base64.b64encode(ulaw).decode()},
        }
        await self._send_json(msg)

    async def _send_mark(self, name: str):
        await self._send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": name},
        })

    async def _send_clear(self):
        await self._send_json({"event": "clear", "streamSid": self.stream_sid})

    # -- Lifecycle ---------------------------------------------------------

    async def start(self):
        self.state = STATE_LISTENING
        self._last_activity = time.monotonic()
        self.history.append({"role": "assistant", "content": config.WELCOME_TEXT})
        await self.speak(config.WELCOME_TEXT)

    async def close(self):
        self._closed = True
        for t in self._turn_tasks:
            if not t.done():
                t.cancel()

    # -- Inbound events ---------------------------------------------------------

    async def on_inbound_media(self, ulaw: bytes):
        """Twilio inbound audio (8kHz μ-law)."""
        if self._closed:
            return
        self._last_activity = time.monotonic()
        pcm16k = ulaw8k_to_pcm16_16k(ulaw)  # pure numpy, microseconds, no thread needed
        is_speech, voiced_ms = self.vad.feed(pcm16k)

        if self.state == STATE_SPEAKING and voiced_ms >= config.BARGE_IN_MS:
            await self._barge_in()
            return
        # Periodic VAD status log to debug "didn't hear anything" issues
        self._vad_log_at = getattr(self, "_vad_log_at", 0.0)
        now = time.monotonic()
        if self.state == STATE_LISTENING and now - self._vad_log_at > 2.0:
            self._vad_log_at = now
            logger.debug("[%s] VAD voiced=%.0fms rms-chunk=%d",
                         self.stream_sid, voiced_ms, len(pcm16k))

        if self.state == STATE_LISTENING:
            # Don't feed ASR while speaking/thinking: keeps the bot's own voice out of history
            await self.asr.feed(pcm16k)

    async def on_mark(self, name: str):
        """Twilio confirms a mark finished playing (or replays unplayed marks after clear)."""
        # Simplified: log only; playback progress advances naturally with the tick cadence
        logger.debug("[%s] mark done: %s", self.stream_sid, name)

    async def on_dtmf(self, digit: str):
        logger.info("[%s] DTMF: %s", self.stream_sid, digit)

    async def on_stop(self):
        await self.close()

    # -- Barge-in -------------------------------------------------------------

    async def _barge_in(self):
        logger.info("[%s] barge-in, clearing buffer", self.stream_sid)
        # _interrupted is owned by the speaking turn loop (on_user_turn):
        # it stops speak() loops and suppresses the history append. Do NOT
        # clear it here — clearing would let the interrupted turn resume
        # sending the rest of its sentences.
        self._interrupted = True
        await self._send_clear()
        # Don't cancel any task: speak / on_user_turn exit on their own via the
        # _interrupted flag; cancel would propagate through receive_text and kill
        # the whole WebSocket.
        # Do NOT reset the VAD: the user is *currently speaking* — wiping the
        # accumulator loses the speech that is already buffered for ASR.
        # Switch to LISTENING immediately: stop feeding the bot's own voice into
        # ASR now (previously it polluted the user's transcript until the turn
        # loop noticed the flag, e.g. "你好。你好，能听见吗？").
        self.state = STATE_LISTENING

    # -- Speak -------------------------------------------------------------

    async def speak(self, text: str):
        """Synthesize and play a text at steady pace; returns early if interrupted.

        TTS generation is faster than real time (RTF<1); pacing sends against an
        absolute clock so the Twilio buffer never overflows and audio plays for
        its true duration. Only one speak may run at a time (turns are naturally
        serial). Sentence history bookkeeping is the caller's job
        (on_user_turn / start).
        """
        if self._closed:
            return
        self.state = STATE_SPEAKING
        self._interrupted = False  # new sentence starts, clear the stale interrupt flag
        t0 = time.monotonic()
        sent = 0
        try:
            async for chunk in synthesize_stream(text):
                if self._interrupted or self._closed:
                    break
                await self._send_media(chunk)
                sent += 1
                # Pace against absolute clock: chunk n goes out at t0 + n*20ms
                ahead = t0 + sent * config.SEND_TICK_SEC - time.monotonic()
                if ahead > 0:
                    await asyncio.sleep(ahead)
        except asyncio.CancelledError:
            logger.info("[%s] speak cancelled after %d chunks", self.stream_sid, sent)
            raise
        except Exception:
            logger.exception("[%s] speak failed", self.stream_sid)
        finally:
            # On barge-in, _barge_in() already switched to LISTENING; don't
            # clobber it here (the interrupting speech must keep flowing to ASR).
            if self.state == STATE_SPEAKING and not self._closed:
                self.state = STATE_LISTENING
        if not self._interrupted and not self._closed:
            logger.info("[%s] spoken %.1fs audio in %.1fs wall",
                        self.stream_sid, sent * 0.02, time.monotonic() - t0)
            self.vad.reset()

    # -- Turn handling ---------------------------------------------------------

    async def handle_turn(self, user_text: str):
        """Task entry for on_user_turn: swallows CancelledError so it never reaches the event loop."""
        try:
            await self.on_user_turn(user_text)
        except asyncio.CancelledError:
            logger.info("[%s] turn cancelled", self.stream_sid)
        except Exception:
            logger.exception("[%s] turn failed", self.stream_sid)

    async def on_user_turn(self, user_text: str):
        """User finished a sentence (triggered by endpointing)."""
        if self._closed or not user_text.strip():
            logger.info("[%s] turn skipped (empty)", self.stream_sid)
            return
        if self.state == STATE_SPEAKING:
            logger.info("[%s] turn skipped (bot still speaking)", self.stream_sid)
            return
        # Barge-in path: the user interrupted the previous reply; their speech
        # since then is real and must be processed — clear the stale flag.
        # (speak() clears it at the start of each sentence, so after the last
        # sentence exits on the flag the value here is stale.)
        self._interrupted = False
        # Prevent concurrent turns: if the previous turn is still winding down, wait for it (max 5s)
        # Note: the current task is itself in _turn_tasks; skip self
        cur = asyncio.current_task()
        for t in list(self._turn_tasks):
            if t is cur or t.done():
                continue
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
        logger.info("[%s] >>> user: %s", self.stream_sid, user_text)
        self.history.append({"role": "user", "content": user_text})
        self.state = STATE_THINKING
        reply_buf = ""
        try:
            async for sent in stream_chat(
                [{"role": "system", "content": config.SYSTEM_PROMPT}, *self.history[-20:]]
            ):
                if self._interrupted or self._closed:
                    break
                reply_buf += sent
                logger.info("[%s] <<< bot: %s", self.stream_sid, sent)
                await self.speak(sent)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] LLM failed", self.stream_sid)
            await self.speak("すみません、ただいま不具合が発生しています。しばらくしてからもう一度おかけ直しください。")
            return
        if not self._closed:
            if self._interrupted:
                # Barge-in: the user is speaking right now. Keep their buffered
                # speech in ASR/VAD so the endpointing loop captures this new
                # turn; don't reset and don't overwrite the state they set.
                return
            # Partial sentences spoken before an error still count as history.
            if reply_buf:
                self.history.append({"role": "assistant", "content": reply_buf})
            await self.asr.reset()
            self.state = STATE_LISTENING
            self.vad.reset()
