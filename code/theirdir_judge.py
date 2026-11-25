"""Judges for the authors' steering test (prereg 2026-11-20 sec. 6).

heuristic : their alias string match (judge-free check, bar B4); used alone in the local smoke.
haiku     : their judge, Claude Haiku 4.5 with their rubric (theirdir_lib.JUDGE_PROMPT), via the
            Message Batches API. Key from env ANTHROPIC_API_KEY (never printed).
The Gemini second judge (B5 kappa sample) is applied after the run, not here.

Each item: dict with keys question, answer, aliases, response. Returns one verdict per item.
"""
from __future__ import annotations

import os
import time

import theirdir_lib as L


def judge_heuristic(items: list[dict]) -> list[str]:
    return ["CORRECT" if L.heuristic_judge(it["response"] or "", it["answer"], it.get("aliases"))
            else "INCORRECT" for it in items]


def _prompt(it: dict) -> str:
    al = it.get("aliases") or []
    return L.JUDGE_PROMPT.format(question=it["question"], answer=it["answer"],
                                 aliases=", ".join(al) if al else "(none)",
                                 response=it["response"] or "")


def judge_haiku_batch(items: list[dict], poll_s: int = 30, log=print) -> list[str]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if len(key) < 20:
        raise RuntimeError("ANTHROPIC_API_KEY missing or too short (shape check only)")
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    out: list[str | None] = [None] * len(items)
    for start in range(0, len(items), 10000):
        chunk = items[start:start + 10000]
        reqs = [{"custom_id": f"i{start + i}",
                 "params": {"model": L.JUDGE_MODEL, "max_tokens": 10,
                            "messages": [{"role": "user", "content": _prompt(it)}]}}
                for i, it in enumerate(chunk)]
        b = client.messages.batches.create(requests=reqs)
        log(f"[judge] batch {b.id}: {len(reqs)} requests")
        while b.processing_status != "ended":
            time.sleep(poll_s)
            b = client.messages.batches.retrieve(b.id)
        for res in client.messages.batches.results(b.id):
            i = int(res.custom_id[1:])
            if res.result.type == "succeeded":
                out[i] = L.parse_verdict(res.result.message.content[0].text)
            else:
                out[i] = "UNCLEAR"
    missing = sum(v is None for v in out)
    if missing:
        raise RuntimeError(f"{missing} judge results missing")
    return out  # type: ignore[return-value]


def judge(items: list[dict], mode: str) -> list[str]:
    if mode == "heuristic":
        return judge_heuristic(items)
    if mode == "haiku":
        return judge_haiku_batch(items)
    raise ValueError(mode)
