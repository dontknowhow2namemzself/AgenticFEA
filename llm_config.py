"""Shared OpenRouter (OpenAI-compatible) configuration for the LLM agent layers.

Loads .env, exposes the model / base-url, and builds an OpenAI client pointed at
OpenRouter. The API key is read ONLY from the environment (.env or the shell) --
never hardcoded, never logged. Reused by the parameterization (set_params), task
understanding, and diagnostics layers.

Iron law reminder: the LLM only fills semantic slots / explains; deterministic
code (schema.py and the physics/verification layers) decides every number.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"


def load_env() -> None:
    """Load .env from the project root. Uses python-dotenv if available, else a
    minimal built-in parser so the only hard dependency is the openai SDK."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
        return
    except ImportError:
        pass
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


load_env()

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.4")
OPENROUTER_EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")


def get_api_key() -> str:
    """Return the OpenRouter API key or raise a clear error if it is missing."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set -- copy .env.example to .env and add your key "
            "(or export OPENROUTER_API_KEY)"
        )
    return key


def make_client():
    """Build an OpenAI client pointed at OpenRouter (raises if openai is absent)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai SDK not installed (pip install openai)") from exc
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_api_key())


def embed_texts(texts: list) -> list:
    """Embed a list of strings via OpenRouter (returns one vector per input).
    Used for RAG over the knowledge base (see vector_store.py)."""
    resp = make_client().embeddings.create(model=OPENROUTER_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
