# SPDX-License-Identifier: Apache-2.0
"""G.711 μ-law codec (pure numpy lookup tables) + 8k→16k upsampling.

Twilio Media Streams audio in both directions is 8kHz μ-law mono; local
models use 16k PCM. The TTS-side 24k→8k resampling lives in tts_engine
(ffmpeg streaming pipeline).
"""

import numpy as np

# ---------------------------------------------------------------------------
# CCITT G.711 μ-law lookup tables (standard ITU-T G.711 recommended table)
# ---------------------------------------------------------------------------

# μ-law byte -> 16-bit linear PCM (signed)
_ULAW_TO_LINEAR = np.zeros(256, dtype=np.int16)

def _build_ulaw_tables():
    for byte in range(256):
        u = ~byte & 0xFF
        sign = u & 0x80
        exp = (u >> 4) & 0x07
        mant = u & 0x0F
        sample = ((mant << 3) + 0x84) << exp
        sample -= 0x84
        if sign:
            sample = -sample
        _ULAW_TO_LINEAR[byte] = np.int16(sample)

_build_ulaw_tables()

# Linear PCM -> μ-law: derived from the decode level table (nearest-level
# selection; matches ffmpeg in practice)
def _build_encode_table():
    # Positive levels decrease with byte value: 0x80 -> 32124 ... 0xFF -> 0
    pos_bytes = np.arange(0x80, 0x100, dtype=np.uint8)
    levels = _ULAW_TO_LINEAR[0x80:0x100].astype(np.int64)
    order = np.argsort(levels)                     # ascending levels
    levels_sorted = levels[order]
    bytes_sorted = pos_bytes[order]
    table = np.zeros(0x8000, dtype=np.uint8)
    pcm = np.arange(0, 0x8000, dtype=np.int64)
    idx = np.clip(np.searchsorted(levels_sorted, pcm), 1, len(levels_sorted) - 1)
    lo, hi = levels_sorted[idx - 1], levels_sorted[idx]
    pick_hi = (hi - pcm) < (pcm - lo)
    table[:] = np.where(pick_hi, bytes_sorted[idx], bytes_sorted[idx - 1])
    return table


_ENCODE_TABLE = _build_encode_table()


def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """int16 PCM -> μ-law bytes (matches the ffmpeg mulaw encoder).

    μ-law is sign-magnitude encoding: positive magnitudes fall in [0x80,0xFF],
    negative magnitudes mirror into [0x00,0x7F].
    """
    pcm = pcm.astype(np.int32)
    mag = np.minimum(np.abs(pcm), 32767)
    pos_byte = _ENCODE_TABLE[mag]
    byte = np.where(pcm < 0, pos_byte ^ 0x80, pos_byte)
    return byte.astype(np.uint8).tobytes()


def ulaw_to_pcm16(ulaw: bytes) -> np.ndarray:
    """μ-law bytes -> int16 PCM。"""
    return _ULAW_TO_LINEAR[np.frombuffer(ulaw, dtype=np.uint8)]


# ---------------------------------------------------------------------------
# 8k -> 16k upsampling (pure numpy linear interpolation, no subprocess)
# ---------------------------------------------------------------------------

def ulaw8k_to_pcm16_16k_np(ulaw: bytes) -> bytes:
    """Twilio 8kHz μ-law -> 16kHz s16le PCM (numpy 2:1 linear-interp upsampling)."""
    pcm = ulaw_to_pcm16(ulaw).astype(np.float32)
    if pcm.size == 0:
        return b""
    n = pcm.size
    # 2:1 upsampling: y[2i]=pcm[i], y[2i+1]=(pcm[i]+pcm[i+1])/2
    y = np.empty(n * 2, dtype=np.float32)
    y[0::2] = pcm
    y[1:-1:2] = (pcm[:-1] + pcm[1:]) * 0.5
    y[-1] = pcm[-1]
    return y.astype(np.int16).tobytes()


def ulaw8k_to_pcm16_16k(ulaw: bytes) -> bytes:
    """Twilio 8kHz μ-law -> 16kHz s16le PCM (feeds ASR). Pure numpy, no subprocess."""
    return ulaw8k_to_pcm16_16k_np(ulaw)


# ---------------------------------------------------------------------------
# Self-test: roundtrip consistency
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sine roundtrip (μ-law should give ~35dB SNR at speech levels)
    t = np.arange(8000) / 8000
    sig = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    back = ulaw_to_pcm16(pcm16_to_ulaw(sig))
    err = np.abs(sig.astype(np.int32) - back.astype(np.int32))
    snr = 20 * np.log10(
        np.sqrt((sig.astype(np.float64) ** 2).mean())
        / np.sqrt((err.astype(np.float64) ** 2).mean())
    )
    print(f"sine roundtrip SNR={snr:.1f} dB max_err={err.max()}")

    # Upsampling smoke test
    pcm16k = ulaw8k_to_pcm16_16k(pcm16_to_ulaw(sig))
    print(f"8k->16k: {len(sig)*2} bytes -> {len(pcm16k)} bytes (expect ~2x)")
