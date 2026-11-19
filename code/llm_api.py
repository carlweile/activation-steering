r"""Send one prompt file to Gemini, OpenAI or Mistral and save the raw reply as JSON, reproducibly.

Usage (from this folder):
  python llm_api.py gemini PROMPT.txt OUT.json [--reps N] [--model ID]
  python llm_api.py openai PROMPT.txt OUT.json [--reps N] [--model ID]
  python llm_api.py mistral PROMPT.txt OUT.json [--reps N] [--model ID]
  python llm_api.py claude PROMPT.txt REPLY.md OUT.json   (wrap a subagent's reply in the same record)
  python llm_api.py smoke gemini|openai|mistral [--model ID]

With --reps N the output path gets _rep1.._repN inserted before ".json" and the calls run
one after another. Every saved JSON carries the SHA-256 of exactly the prompt text sent, so a
score can always be tied back to the bytes the judge saw.

Mistral (the author, 18.11.2026): default mistral-large-4 with reasoning_effort
"high", reply streamed through sse(). Without reasoning_effort Mistral thinks 0 tokens, and
magistral-medium-latest now aliases a plain Mistral Medium; long non-streamed thinking calls died at 30 min.

Keys: environment variables GEMINI_API_KEY, OPENAI_API_KEY and MISTRAL_API_KEY. They are checked by
SHAPE only (exists, length, not a placeholder) and are never printed, logged or saved.
Only the Python standard library is used.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator

MODELS = {"gemini": "gemini-3.1-pro-preview", "openai": "gpt-6-astra", "mistral": "mistral-large-4", "claude": "claude-fable-5-1"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
OPENAI_URL = "https://api.openai.com/v1/responses"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_REASONING_EFFORT = "high"  # without it the Mistral seat does not think at all (0 thinking tokens)
STREAM_IDLE_S = 600                # a stream that sends nothing for 10 minutes is a failure
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "mistral": "MISTRAL_API_KEY"}
TERMINAL = {"completed", "incomplete", "failed", "cancelled"}
RETRY_CODES = {429, 500, 502, 503, 504}


class ApiError(Exception):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body}")
        self.code = code


class StreamError(Exception):
    """A streamed reply broke off after the call went out: nothing came for the idle time, or the stream sent an
    error event. The message is this module's own words, never the provider's text, so a caller may keep it."""


def read_key(provider: str) -> str:
    name = KEY_ENV[provider]
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"environment variable {name} is not set")
    if len(key) < 20 or key.startswith("PASTE"):
        raise SystemExit(f"{name} fails the shape check (length {len(key)})")
    return key


def _http(url: str, headers: dict, body: bytes | None = None, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Error bodies never echo request headers, so the key cannot appear here.
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace")[:1500]) from None


def _events(lines: Iterable[bytes]) -> Iterator[dict]:
    """The JSON object on each `data:` line of a server-sent-event stream. Blank lines, comments, other fields
    and the closing "[DONE]" are passed over; an error event raises."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
            continue
        event = json.loads(line[5:])
        if not isinstance(event, dict):
            raise StreamError("the stream sent an event that is not a JSON object")
        if event.get("error"):
            code = event["error"].get("code") if isinstance(event["error"], dict) else None
            raise StreamError("the stream sent an error event"
                              + (f" (code {code})" if isinstance(code, int) and not isinstance(code, bool) else ""))
        yield event


def sse(url: str, headers: dict, body: bytes, idle_timeout: float = STREAM_IDLE_S) -> Iterator[dict]:
    """POST `body` and yield each event of the streamed reply (server-sent events), in order.

    Streaming keeps bytes flowing while a model thinks for a long time. `idle_timeout` is the longest silence
    allowed between two reads, not a limit on the whole reply. An HTTP error before the first event is ApiError;
    a busy or rate-limited provider (RETRY_CODES) is asked again first."""
    for attempt in range(4):
        request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=idle_timeout)
            break
        except urllib.error.HTTPError as exc:
            # Error bodies never echo request headers, so the key cannot appear here.
            error = ApiError(exc.code, exc.read().decode("utf-8", "replace")[:1500])
            if exc.code not in RETRY_CODES or attempt == 3:
                raise error from None
            print(f"stream HTTP {exc.code}, retry in {30 * (attempt + 1)} s", flush=True)
            time.sleep(30 * (attempt + 1))
        except TimeoutError:
            raise StreamError(f"the provider sent nothing for {idle_timeout:g} s") from None
    with response:
        try:
            yield from _events(response)
        except TimeoutError:
            raise StreamError(f"the stream sent nothing for {idle_timeout:g} s") from None


def _latest(meta: dict, **seen: Any) -> dict:
    """A streamed reply's meta so far: per key, the value of the newest event that carries one (the usage, with
    the thinking tokens in completion_tokens, comes in the last event)."""
    return {name: value or meta.get(name) for name, value in seen.items()}


def call_gemini(prompt: str, key: str, model: str, timeout: int = 900) -> tuple[str, dict]:
    headers = {"x-goog-api-key": key, "Content-Type": "application/json; charset=utf-8"}
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}]}).encode("utf-8")
    for attempt in range(4):
        try:
            resp = _http(GEMINI_URL.format(m=model), headers, body, timeout)
            break
        except ApiError as exc:
            if exc.code not in RETRY_CODES or attempt == 3 or "limit: 0" in str(exc):
                raise
            print(f"gemini {exc.code}, retry in {30 * (attempt + 1)} s", flush=True)
            time.sleep(30 * (attempt + 1))
    cand = (resp.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
    meta = {"usage": resp.get("usageMetadata"), "model_version": resp.get("modelVersion"),
            "finish_reason": cand.get("finishReason"), "response_id": resp.get("responseId")}
    return text.strip(), meta


def call_openai(prompt: str, key: str, model: str, timeout_s: int = 3600) -> tuple[str, dict]:
    """Background mode + polling. No reasoning
    parameter on purpose: run 2 (2026-09-18) called gpt-6-astra with model defaults, and the
    new scores must stay comparable with those."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json; charset=utf-8"}
    body = json.dumps({"model": model, "input": prompt, "background": True,
                       "store": True}).encode("utf-8")
    resp = _http(OPENAI_URL, headers, body, 120)
    rid = resp.get("id")
    print(f"openai submitted id={rid} status={resp.get('status')}", flush=True)
    deadline, failures = time.time() + timeout_s, 0
    while resp.get("status") not in TERMINAL:
        if time.time() > deadline:
            raise SystemExit(f"gave up after {timeout_s} s; {rid} still {resp.get('status')}")
        time.sleep(15)
        try:
            resp = _http(f"{OPENAI_URL}/{rid}", headers, None, 60)
            failures = 0
        except (ApiError, TimeoutError, urllib.error.URLError) as exc:
            failures += 1
            if failures >= 10:
                raise SystemExit(f"10 poll failures in a row for {rid}: {exc}")
    if resp.get("status") in {"failed", "cancelled"}:
        raise SystemExit(f"{rid} ended {resp.get('status')}: {resp.get('error')}")
    chunks = [c.get("text", "") for item in resp.get("output", []) if item.get("type") == "message"
              for c in item.get("content", []) if c.get("type") in ("output_text", "text")]
    meta = {"usage": resp.get("usage"), "model_version": resp.get("model"),
            "status": resp.get("status"), "response_id": rid}
    return "\n".join(chunks).strip(), meta


def call_mistral(prompt: str, key: str, model: str, timeout: float = STREAM_IDLE_S,
                 reasoning_effort: str | None = MISTRAL_REASONING_EFFORT) -> tuple[str, dict]:
    """EU third voice (the author 22.10.2026: no China-hosted models, no Grok). Streamed, with `reasoning_effort`;
    `timeout` is the idle time of sse(). A thinking model sends its content as a list of thinking and text
    chunks; only the text is kept."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json; charset=utf-8"}
    request: dict = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    text: list[str] = []
    meta: dict = {}
    for event in sse(MISTRAL_URL, headers, json.dumps(request).encode("utf-8"), timeout):
        choice = (event.get("choices") or [{}])[0]
        content = (choice.get("delta") or {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text") or "" for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
        text.append(content if isinstance(content, str) else "")
        meta = _latest(meta, usage=event.get("usage"), model_version=event.get("model"),
                       finish_reason=choice.get("finish_reason"), response_id=event.get("id"))
    return "".join(text).strip(), meta


CALLERS = {"gemini": call_gemini, "openai": call_openai, "mistral": call_mistral}


def record(provider: str, model: str, prompt: str, text: str, meta: dict, started: str,
           seconds: float, out: Path) -> dict:
    rec = {"provider": provider, "model": model,
           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
           "prompt_chars": len(prompt), "started": started, "seconds": round(seconds, 1),
           "text": text, **meta}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(text):,} chars, model {meta.get('model_version') or model})", flush=True)
    return rec


def run_once(provider: str, model: str, prompt: str, out: Path) -> dict:
    started, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.time()
    text, meta = CALLERS[provider](prompt, read_key(provider), model)
    return record(provider, model, prompt, text, meta, started, time.time() - t0, out)


def main(argv: list[str]) -> int:
    model_override = None
    if "--model" in argv:
        i = argv.index("--model")
        model_override = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "smoke":
        provider = argv[1]
        model = model_override or MODELS[provider]
        text, meta = CALLERS[provider]("Reply with exactly the two letters: OK", read_key(provider), model)
        print(f"{provider} {model} -> {text!r} | version {meta.get('model_version')} | usage {meta.get('usage')}")
        return 0 if "OK" in text else 1
    if cmd == "claude":
        prompt, reply, out = (Path(a) for a in argv[1:4])
        text = reply.read_text(encoding="utf-8").strip()
        meta = {"model_version": MODELS["claude"], "note": "fresh Claude Code subagent, text only, saved verbatim"}
        record("claude", MODELS["claude"], prompt.read_text(encoding="utf-8"), text, meta,
               time.strftime("%Y-%m-%d %H:%M:%S"), 0.0, out)
        return 0
    if cmd not in CALLERS or len(argv) < 3:
        raise SystemExit(__doc__)
    prompt_path, out = Path(argv[1]), Path(argv[2])
    reps = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else 1
    model = model_override or MODELS[cmd]
    prompt = prompt_path.read_text(encoding="utf-8")
    for k in range(1, reps + 1):
        target = out if reps == 1 else out.with_name(f"{out.stem}_rep{k}{out.suffix}")
        if target.exists():
            print(f"skip {target} (exists)", flush=True)
            continue
        run_once(cmd, model, prompt, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
