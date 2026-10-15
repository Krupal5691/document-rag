"""Gemini API wrapper: text generation, JSON generation, vision, embeddings.
Retries on rate limits; responses for deterministic calls are cached on disk."""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from . import config

EMBED_BATCH = 100


class MissingKeyError(RuntimeError):
    pass


def has_key() -> bool:
    return bool(config.GEMINI_API_KEY)


@lru_cache(maxsize=1)
def client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise MissingKeyError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _with_retry(fn, attempts: int = 6):
    for i in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i * 3))     # free tier: back off on per-minute limits


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

def generate(prompt: str, system: str, model: str | None = None, temperature: float = 0.0) -> str:
    model = model or config.GEN_MODEL
    resp = _with_retry(lambda: client().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
    ))
    return (resp.text or "").strip()


def generate_json(prompt: str, system: str, schema: dict, model: str | None = None, cache: str | None = None) -> dict:
    """Structured output: the model must return JSON matching `schema`. Cached when `cache` is set."""
    model = model or config.QUERY_MODEL

    def call():
        resp = _with_retry(lambda: client().models.generate_content(
            model=model, contents=prompt,
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
    resp = _with_retry(lambda: client().models.generate_content(
        model=model,
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
        resp = _with_retry(lambda: client().models.embed_content(
            model=config.GEMINI_EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.GEMINI_EMBED_DIM),
        ))
        vectors.extend(e.values for e in resp.embeddings)
    return vectors
