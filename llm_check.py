#!/usr/bin/env python3
"""One-time LLM connectivity check for the OpenRouter endpoint.

The agent-layer equivalent of check_env.py: confirms the openai SDK is installed,
the API key is present (.env or shell), and a minimal chat call to the configured
model returns a reply. Run this before building the set_params Tool.

A successful call (no exception) means connectivity is good; the "PONG" prompt is
just so you can eyeball the model's reply. The key is never printed.

Usage:
    python llm_check.py
"""

import sys

import llm_config


def main() -> int:
    print("=" * 64)
    print("LLM connectivity check (OpenRouter)")
    print("=" * 64)
    print(f"base_url : {llm_config.OPENROUTER_BASE_URL}")
    print(f"model    : {llm_config.OPENROUTER_MODEL}")

    # API key presence (length only -- never print key material).
    try:
        key = llm_config.get_api_key()
        print(f"api_key  : present (length {len(key)})")
    except Exception as exc:  # noqa: BLE001
        print(f"api_key  : MISSING -> {exc}")
        return 1

    # Build the client (checks the openai SDK is installed).
    try:
        client = llm_config.make_client()
    except Exception as exc:  # noqa: BLE001
        print(f"client   : FAILED -> {exc}")
        return 1

    print("calling the model with a tiny prompt ...")
    try:
        resp = client.chat.completions.create(
            model=llm_config.OPENROUTER_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        )
        reply = resp.choices[0].message.content
        print(f"reply    : {reply!r}")
        try:
            print(f"usage    : {resp.usage}")
        except Exception:  # noqa: BLE001
            pass
        print("-" * 64)
        print("RESULT: PASS (OpenRouter reachable, model responded)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"call     : FAILED -> {exc.__class__.__name__}: {exc}")
        print("-" * 64)
        print("RESULT: FAIL (check the key, model slug, network, or OpenRouter credits)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
