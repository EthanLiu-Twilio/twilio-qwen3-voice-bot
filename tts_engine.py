# SPDX-License-Identifier: Apache-2.0
"""TTS engine: calls local oMLX `/v1/audio/speech` (stream=true, wav).

oMLX runs mlx-audio Qwen3-TTS internally (24kHz WAV stream output); this
process loads no model. Continuously resampled to 8kHz via ffmpeg and emitted
as μ-law blocks aligned to 160 bytes (20ms). The Base model supports voice
cloning via ref_audio + ref_text (paths configured in .env, uploaded base64).
"""

import asyncio
import base64
import logging
import os
import select
import subprocess
import threading
import wave
from pathlib import Path

import httpx
import numpy as np

import config
from audio_codec import pcm16_to_ulaw

logger = logging.getLogger(__name__)

# 20ms @ 8kHz: s16 mono = 320 bytes PCM, μ-law = 160 bytes
_PCM_BYTES_PER_TICK_8K = 320
# Minimum input granularity fed to ffmpeg (~66ms @ 24kHz s16); batch before
# feeding to reduce call count
_INPUT_CHUNK = 3200


_LANG_PREFIXES = {
    "zh": ("你好", "您", "什么", "谢谢", "请问", "的", "了", "吗"),
    "ja": ("もしもし", "です", "ます", "ご", "の", "は", "を", "か"),
    "en": ("hello", "the", "you", "please", "thank", "this", "what"),
}


def detect_lang(text: str) -> str:
    """Heuristic dominant-language detection by charset + keywords: zh / ja / en."""
    if not text:
        return "ja"
    # Any hiragana/katakana present => Japanese (Chinese text has no kana)
    kana = sum(1 for c in text if "぀" <= c <= "ゟ" or "゠" <= c <= "ヿ")
    han = sum(1 for c in text if "一" <= c <= "鿿")
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if kana >= 2:
        return "ja"
    if han >= 2:
        return "zh"
    if ascii_letters >= 3:
        return "en"
    low = text.lower()
    for lang, kws in _LANG_PREFIXES.items():
        if any(k in low for k in kws):
            return lang
    return "ja"


def _load_ref_audio_b64(lang: str = "ja") -> tuple[str | None, str]:
    """Load the reference audio for a language (to 24kHz s16 wav then base64); cached per language."""
    global _REF_CACHE
    if lang in _REF_CACHE:
        return _REF_CACHE[lang]
    path = config.REF_AUDIO_PATHS.get(lang) or config.REF_AUDIO_PATH
    ref_text = config.REF_TEXTS.get(lang) or config.REF_TEXT
    result: tuple[str | None, str] = (None, "")
    if path and Path(path).expanduser().is_file() and ref_text:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", str(Path(path).expanduser()),
             "-f", "wav", "-ar", str(config.TTS_SAMPLE_RATE), "-ac", "1",
             "pipe:1"],
            capture_output=True,
        )
        if proc.returncode != 0:
            logger.error("ref_audio(%s) decode failed: %s", lang, proc.stderr.decode()[:300])
        else:
            result = (base64.b64encode(proc.stdout).decode(), ref_text)
            logger.info("ref_audio(%s) loaded (%d bytes wav b64)", lang, len(proc.stdout))
    else:
        logger.warning("REF_AUDIO_PATH(%s) unset or file missing; running without reference audio", lang)
    _REF_CACHE[lang] = result
    return result


_REF_CACHE: dict[str, tuple[str | None, str]] = {}


class _StreamResampler:
    """Streaming ffmpeg resampler: 24kHz s16le in, 8kHz s16le out; continuous across blocks, no boundary noise."""

    def __init__(self, in_rate: int):
        self._proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(in_rate), "-ac", "1", "-i", "pipe:0",
             "-f", "s16le", "-ar", str(config.TWILIO_SAMPLE_RATE), "-ac", "1",
             "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._in_lock = threading.Lock()
        self._leftover = bytearray()

    def feed(self, pcm: bytes) -> list[bytes]:
        """Feed input PCM; returns all 20ms output blocks currently available (non-blocking)."""
        with self._in_lock:
            self._proc.stdin.write(pcm)
            self._proc.stdin.flush()
        self._drain()
        out = []
        while len(self._leftover) >= _PCM_BYTES_PER_TICK_8K:
            out.append(bytes(self._leftover[:_PCM_BYTES_PER_TICK_8K]))
            del self._leftover[:_PCM_BYTES_PER_TICK_8K]
        return out

    def _drain(self):
        """Drain the stdout pipe until empty (non-blocking)."""
        while True:
            r, _, _ = select.select([self._proc.stdout], [], [], 0)
            if not r:
                return
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                return
            self._leftover.extend(chunk)

    def flush(self) -> list[bytes]:
        """Close stdin and collect all remaining output, cut into 20ms blocks (tail zero-padded)."""
        with self._in_lock:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        try:
            for chunk in iter(lambda: os.read(self._proc.stdout.fileno(), 65536), b""):
                self._leftover.extend(chunk)
        except OSError:
            pass
        self._proc.wait(timeout=10)
        out = []
        while len(self._leftover) >= _PCM_BYTES_PER_TICK_8K:
            out.append(bytes(self._leftover[:_PCM_BYTES_PER_TICK_8K]))
            del self._leftover[:_PCM_BYTES_PER_TICK_8K]
        if self._leftover:
            tail = bytes(self._leftover) + b"\x00" * (_PCM_BYTES_PER_TICK_8K - len(self._leftover))
            out.append(tail)
            self._leftover.clear()
        return out

    def kill(self):
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass


def _to_ulaw_ticks(pcm8k: bytes):
    """Cut 8kHz s16 PCM into 20ms blocks and convert to μ-law."""
    buf = bytearray(pcm8k)
    while len(buf) >= _PCM_BYTES_PER_TICK_8K:
        blk = np.frombuffer(bytes(buf[:_PCM_BYTES_PER_TICK_8K]), dtype=np.int16)
        yield pcm16_to_ulaw(blk)
        del buf[:_PCM_BYTES_PER_TICK_8K]


async def synthesize_stream(text: str):
    """Stream-synthesize a text.

    Yields 8kHz μ-law bytes (160 bytes / 20ms aligned).
    Cancelling this coroutine closes the HTTP stream and stops generation on oMLX.
    """
    lang = detect_lang(text)
    ref_b64, ref_text = await asyncio.to_thread(_load_ref_audio_b64, lang)

    payload: dict = {
        "model": config.TTS_MODEL_ID,
        "input": text,
        "stream": True,
        "response_format": "wav",
        "streaming_interval": 0.4,
        "speed": config.TTS_SPEED,
    }
    if ref_b64:
        payload["ref_audio"] = ref_b64
        payload["ref_text"] = ref_text

    resampler: _StreamResampler | None = None
    buf = bytearray()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0)) as client:
            async with client.stream(
                "POST",
                f"{config.OMLX_BASE_URL}/audio/speech",
                json=payload,
                headers={"Authorization": f"Bearer {config.OMLX_API_KEY}"},
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeError(f"TTS HTTP {resp.status_code}: {body}")

                async for chunk in resp.aiter_bytes():
                    if resampler is None:
                        buf.extend(chunk)
                        if len(buf) < 44:
                            continue
                        resampler = _StreamResampler(config.TTS_SAMPLE_RATE)
                        chunk = bytes(buf[44:])  # strip the 44-byte WAV header once
                        buf.clear()
                        if not chunk:
                            continue

                    buf.extend(chunk)
                    while len(buf) >= _INPUT_CHUNK:
                        in_block = bytes(buf[:_INPUT_CHUNK])
                        del buf[:_INPUT_CHUNK]
                        ticks = await asyncio.to_thread(resampler.feed, in_block)
                        for tick in ticks:
                            yield pcm16_to_ulaw(np.frombuffer(tick, dtype=np.int16))

        # HTTP stream ended: flush remaining input and output
        if resampler:
            if buf:
                ticks = await asyncio.to_thread(resampler.feed, bytes(buf))
                buf.clear()
                for tick in ticks:
                    yield pcm16_to_ulaw(np.frombuffer(tick, dtype=np.int16))
            for tick in await asyncio.to_thread(resampler.flush):
                yield pcm16_to_ulaw(np.frombuffer(tick, dtype=np.int16))
            resampler = None
    finally:
        if resampler:
            resampler.kill()


async def synthesize_to_wav(text: str, out_path: str):
    """Debug helper: synthesize and save a wav file."""
    from audio_codec import ulaw_to_pcm16

    frames = bytearray()
    async for chunk in synthesize_stream(text):
        frames.extend(chunk)
    pcm = ulaw_to_pcm16(bytes(frames))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.TWILIO_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    logger.info("saved %s (%.1fs audio)", out_path, len(pcm) / config.TWILIO_SAMPLE_RATE)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，这是一段语音合成测试。"
    asyncio.run(synthesize_to_wav(text, "/tmp/tts_test.wav"))
    print("wrote /tmp/tts_test.wav")
