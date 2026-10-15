"""Gemini API wrapper: text generation, JSON generation, vision, embeddings.
Retries on rate limits; responses for deterministic calls are cached on disk."""
from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache

import httpx
from google import genai
from google.genai import errors, types

from . import config

EMBED_BATCH = 100


class MissingKeyError(RuntimeError):
    pass


class QuotaExhaustedError(RuntimeError):
    """Daily quota used up: retrying in a few seconds will not help."""


def _long_wait(e: errors.APIError) -> bool:
    """True when the API asks us to wait minutes or hours (a daily quota), not seconds."""
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)?s?", str(e))
    if not m:
        return "PerDay" in str(e)
    hours, minutes = int(m.group(1) or 0), int(m.group(2) or 0)
    return hours > 0 or minutes >= 2


def has_key() -> bool:
    return bool(config.GEMINI_API_KEY)


@lru_cache(maxsize=1)
def client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise MissingKeyError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _with_retry(fn, attempts: int = 6):
    """Retry rate limits (429) patiently - the free tier has per-minute quotas. Give up quickly on
    overload (503) so the caller can fall back to another model."""
    for i in range(attempts):
        try:
            return fn()
        except httpx.TransportError:                # dropped connection, DNS hiccup, timeout
            if i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i * 5))
        except errors.APIError as e:
            if e.code == 429 and _long_wait(e):
                raise QuotaExhaustedError(str(e)[:300]) from e
            retryable = e.code in (429, 500, 502, 504) or (e.code == 503 and i < 2)
            if not retryable or i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i * 5))


EXHAUSTED: set[str] = set()      # models whose daily quota ran out in this process
LAST_MODEL: str | None = None    # model that produced the most recent response


def _with_fallback(model: str, call):
    """call(model) with the requested model, then config.FALLBACK_MODELS when it is overloaded (503),
    retired (404) or out of daily quota. The free tier allows ~20 requests per model per day, so
    rotating models keeps a long evaluation running."""
    global LAST_MODEL
    last: Exception | None = None
    for m in [model] + [f for f in config.FALLBACK_MODELS if f != model]:
        if m in EXHAUSTED:
            continue
        try:
            resp = _with_retry(lambda m=m: call(m))
            LAST_MODEL = m
            return resp
        except QuotaExhaustedError as e:
            EXHAUSTED.add(m)
            last = e
        except errors.APIError as e:
            if e.code not in (404, 503):
                raise
            last = e
    raise QuotaExhaustedError(f"All models unavailable or out of quota ({sorted(EXHAUSTED)}): {last}")


# ------------------------------------------------------------------------------------ cache

def _cache_path(kind: str, key: dict):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
    path = config.CACHE_DIR / "llm" / kind / f"{digest}.json"
    return path


def _cached(kind: str, key: dict, compute):
    path = _cache_path(kind, key)
    if path.exists():
        return json.loads(path.read_text())["value"]
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "value": value}, indent=1))
    return value


# ------------------------------------------------------------------------------- generation

def generate(prompt: str, system: str, model: str | None = None, temperature: float = 0.0,
             cache: str | None = None) -> str:
    """Plain-text answer. With `cache`, identical requests are served from disk (used by the evaluator
    so reruns cost no quota)."""
    global LAST_MODEL
    model = model or config.GEN_MODEL

    def call():
        resp = _with_fallback(model, lambda m: client().models.generate_content(
            model=m, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        ))
        return {"text": (resp.text or "").strip(), "model": LAST_MODEL}

    if cache:
        out = _cached(cache, {"model": model, "system": system, "prompt": prompt}, call)
        LAST_MODEL = out["model"]
        return out["text"]
    return call()["text"]


def generate_json(prompt: str, system: str, schema: dict, model: str | None = None, cache: str | None = None) -> dict:
    """Structured output: the model must return JSON matching `schema`. Cached when `cache` is set."""
    model = model or config.QUERY_MODEL

    def call():
        resp = _with_fallback(model, lambda m: client().models.generate_content(
            model=m, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.0,
                response_mime_type="application/json", response_schema=schema,
            ),
        ))
        return json.loads(resp.text or "{}")

    if cache:
        return _cached(cache, {"model": model, "system": system, "prompt": prompt}, call)
    return call()


def describe_image(png: bytes, prompt: str, schema: dict, model: str | None = None) -> dict:
    model = model or config.VISION_MODEL
    resp = _with_fallback(model, lambda m: client().models.generate_content(
        model=m,
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json", response_schema=schema,
        ),
    ))
    return json.loads(resp.text or "{}")


# ------------------------------------------------------------------------------- embeddings

def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = _with_retry(lambda batch=batch: client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.GEMINI_EMBED_DIM),
        ))
        vectors.extend(e.values for e in resp.embeddings)
    return vectors
