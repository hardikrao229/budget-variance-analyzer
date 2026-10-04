"""
llm.py - thin, defensive wrapper around Google's Gemini API (free tier).

Design goals (these are referenced in the project document):
  * Key comes from Streamlit secrets / env var, or is pasted by the user in the sidebar.
    It is never written to disk or logged.
  * Low temperature for repeatable, business-style output.
  * Model fallback chain: if the preferred model is deprecated / unavailable,
    try the next one instead of crashing.
  * Every call returns (text, meta) - never raises. The app decides what to do
    when meta["ok"] is False (it switches to a rule-based fallback).
  * Simple per-session rate guard so one user can't burn the free quota.
"""
from __future__ import annotations

import json
import os
import re
import time

import streamlit as st

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]
MIN_SECONDS_BETWEEN_CALLS = 2.0


def get_api_key() -> str | None:
    """Look for a key in (1) Streamlit secrets, (2) env var, (3) the sidebar box."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("_user_api_key") or None
    return key


def sidebar_key_box() -> None:
    """Show a key box only when the deployer has not configured one."""
    has_server_key = False
    try:
        has_server_key = bool(st.secrets.get("GEMINI_API_KEY"))  # type: ignore[attr-defined]
    except Exception:
        pass
    has_server_key = has_server_key or bool(os.environ.get("GEMINI_API_KEY"))
    with st.sidebar:
        st.markdown("### AI engine")
        if has_server_key:
            st.success("Gemini connected (server key)")
        else:
            st.text_input(
                "Gemini API key",
                type="password",
                key="_user_api_key",
                help="Free key from aistudio.google.com. Kept only in this browser session.",
            )
            if not st.session_state.get("_user_api_key"):
                st.info("No key: the app runs in rule-based offline mode.")
        st.toggle(
            "Force offline mode (demo the fallback)",
            key="_force_offline",
            help="Simulates the AI API being down.",
        )


def _models() -> list[str]:
    preferred = None
    try:
        preferred = st.secrets.get("GEMINI_MODEL")  # type: ignore[attr-defined]
    except Exception:
        pass
    chain = ([preferred] if preferred else []) + DEFAULT_MODELS
    seen, out = set(), []
    for m in chain:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def ai_available() -> bool:
    return bool(get_api_key()) and not st.session_state.get("_force_offline", False)


def generate(
    prompt: str | list,
    system: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    """Call Gemini. `prompt` may be a string or a list of {"role","text"} turns.
    Returns (text, meta). Never raises."""
    meta = {"ok": False, "model": None, "error": None, "latency_s": None}
    if st.session_state.get("_force_offline"):
        meta["error"] = "Offline mode forced by user"
        return "", meta
    key = get_api_key()
    if not key:
        meta["error"] = "No API key configured"
        return "", meta

    # crude per-session rate guard (free tier is ~10 requests/minute)
    last = st.session_state.get("_last_call_ts", 0.0)
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - last)
    if wait > 0:
        time.sleep(wait)

    try:
        from google import genai
        from google.genai import types
    except Exception as e:  # pragma: no cover
        meta["error"] = f"google-genai not installed: {e}"
        return "", meta

    if isinstance(prompt, str):
        contents = prompt
    else:
        contents = [
            types.Content(
                role="model" if t["role"] in ("assistant", "model") else "user",
                parts=[types.Part(text=t["text"])],
            )
            for t in prompt
        ]

    cfg = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        cfg["response_mime_type"] = "application/json"

    client = genai.Client(api_key=key)
    last_err = None
    for model in _models():
        for attempt in range(2):  # one retry for transient errors
            t0 = time.time()
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg),
                )
                st.session_state["_last_call_ts"] = time.time()
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError("Empty response (possibly blocked by safety filter)")
                meta.update(ok=True, model=model, latency_s=round(time.time() - t0, 2))
                return text, meta
            except Exception as e:
                last_err = str(e)
                msg = last_err.lower()
                if ("not found" in msg or "404" in msg or "not supported" in msg
                        or "no longer available" in msg):
                    break  # model retired / unavailable -> try next model
                if "denied access" in msg:
                    meta["error"] = ("Google has blocked this key's project ('project denied access') - "
                                     "create a key from a different Google account")
                    return "", meta
                if "api key" in msg or "401" in msg:
                    meta["error"] = "API key rejected - check the key"
                    return "", meta
                if "permission" in msg or "403" in msg:
                    break  # this model not allowed for the project -> try next model
                if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                    time.sleep(3)
                else:
                    time.sleep(1)
    meta["error"] = (last_err or "Unknown error")[:300]
    return "", meta


def parse_json(text: str):
    """Tolerant JSON parse: strips ``` fences and grabs the outermost object/array."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def ai_badge(meta: dict) -> None:
    if meta.get("ok"):
        st.caption(f"Generated by {meta['model']} in {meta['latency_s']}s - AI output, verify before acting.")
    else:
        st.warning(
            f"AI unavailable ({meta.get('error')}). Showing rule-based output instead.",
            icon="⚠️",
        )
