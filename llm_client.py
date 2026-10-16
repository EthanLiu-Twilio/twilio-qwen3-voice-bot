# SPDX-License-Identifier: Apache-2.0
"""LLM client: streaming calls to oMLX /v1/chat/completions, split output by sentence."""

import json
import logging
import re
from typing import AsyncIterator

import httpx

import config

logger = logging.getLogger(__name__)

# Sentence-ending punctuation (Chinese + English)
_SENT_END = re.compile(r"[。！？；.!?;]")


def split_sentences(buf: str) -> tuple[list[str], str]:
    """Cut complete sentences from the buffer; returns (sentences, remainder)."""
    sents: list[str] = []
    last = 0
    for m in _SENT_END.finditer(buf):
        sent = buf[last:m.end()].strip()
        if sent:
            sents.append(sent)
        last = m.end()
    return sents, buf[last:]


async def stream_chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> AsyncIterator[str]:
    """Streaming chat; yields text one sentence at a time."""
    url = f"{config.OMLX_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.OMLX_API_KEY:
        headers["Authorization"] = f"Bearer {config.OMLX_API_KEY}"

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    buf = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, ValueError):
                    continue
                if not delta:
                    continue
                buf += delta
                sents, buf = split_sentences(buf)
                for s in sents:
                    yield s
    if buf.strip():
        yield buf.strip()
