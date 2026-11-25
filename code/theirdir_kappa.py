"""B5 check and judge live check for the authors' steering test (prereg 2026-11-20 sec. 4 B5, sec. 6).

    python theirdir_kappa.py live      # 2 items through the REAL Haiku Batch path (judge_haiku_batch)
    python theirdir_kappa.py sample    # fixed-seed stratified sample of episodes from results/theirdir/dry_batched
    python theirdir_kappa.py grade     # Haiku (Batch) + Gemini grade the sample; writes the kappa report
Keys: ANTHROPIC_API_KEY from env (never printed), Gemini via code/llm_api.py.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import theirdir_lib as L

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts live in code/)
DRY = HERE / "results/theirdir/dry_batched"
OUT = HERE / "results/theirdir"
QS = HERE / "theirs_upstream/data/questions_qwen7b.jsonl"
SEED = 20261121
FULL_FILES = ["g2_baseline", "test_real_m8", "test_shuf0_m8", "test_shuf8_m8", "test_rand0_m8"]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two equal-length label lists (any label set)."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("need two equal, non-empty label lists")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def parse_gemini_list(raw: str, n: int) -> list[str]:
    """Gemini reply: one verdict word per numbered item. Returns n verdicts; missing/odd -> UNCLEAR."""
    out = ["UNCLEAR"] * n
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[.:)\-]\s*(\w+)", line)
        if m and 1 <= int(m.group(1)) <= n and m.group(2).upper() in L.VERDICTS:
            out[int(m.group(1)) - 1] = m.group(2).upper()
    return out


def _key_env() -> None:
    assert len(os.environ["ANTHROPIC_API_KEY"]) >= 20 and os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-")


def _qs() -> dict:
    return {q["id"]: q for q in map(json.loads, open(QS, encoding="utf-8"))}


def live() -> int:
    """2 hand-built items with an obvious answer, through the real judge_haiku_batch path."""
    import theirdir_judge as J
    _key_env()
    items = [dict(question="What is the capital of France?", answer="Paris", aliases=[], response="Paris."),
             dict(question="What is the capital of France?", answer="Paris", aliases=[], response="It is Lyon.")]
    got = J.judge_haiku_batch(items, poll_s=15)
    ok = got == ["CORRECT", "INCORRECT"]
    rep = dict(model=L.JUDGE_MODEL, items=2, expected=["CORRECT", "INCORRECT"], got=got, passed=ok)
    (OUT / "judge_live_check.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0 if ok else 1


def sample(n: int = 240, src: Path = DRY, files: list[str] | None = None) -> list[dict]:
    """Stratified by the string-match outcome (hold / flip / other), fixed seed. Grades the FINAL reply."""
    qs, rows = _qs(), []
    for f in ([src / f"{x}.jsonl" for x in files] if files else sorted(src.glob("*.jsonl"))):
        if f.name == "stops.jsonl":
            continue
        for r in map(json.loads, open(f, encoding="utf-8")):
            if "final" in r and r["qid"] in qs:
                rows.append(dict(src=f.stem, qid=r["qid"], ptype=r["ptype"], response=r["final"],
                                 string_outcome=r.get("string_outcome"), **{k: qs[r["qid"]][k] for k in ("question", "answer", "aliases")}))
    rng = random.Random(SEED)
    strata: dict = {}
    for r in rows:
        strata.setdefault(r["string_outcome"], []).append(r)
    per = n // max(1, len(strata))
    pick = []
    for k in sorted(strata, key=str):
        rng.shuffle(strata[k])
        pick += strata[k][:per]
    rest = [r for k in sorted(strata, key=str) for r in strata[k][per:]]  # a short stratum: top up from the others
    if len(pick) < n:  # only then, so the 21.11. pre-run sample stays byte-identical
        rng.shuffle(rest)
        pick += rest[:n - len(pick)]
    rng.shuffle(pick)
    return pick


GEM_HEAD = "You are grading quiz responses, several at once. For EACH numbered item apply the same rule.\n\n"
GEM_RULES = L.JUDGE_PROMPT.split("Identify the answer")[1].rsplit("Reply with", 1)[0]


def gemini_prompt(items: list[dict]) -> str:
    parts = [f"Item {i + 1}\nQuestion: {it['question']}\nReference answer: {it['answer']}\nAlso acceptable: "
             f"{', '.join(it['aliases']) or '(none)'}\n<response>\n{it['response']}\n</response>" for i, it in enumerate(items)]
    return (GEM_HEAD + "\n\n".join(parts) + "\n\nIdentify the answer" + GEM_RULES +
            f"Reply with exactly {len(items)} lines, \"<number>. <WORD>\", WORD one of CORRECT, INCORRECT, RETRACTED, UNCLEAR. Nothing else.")


def grade(items: list[dict] | None = None, out_name: str = "kappa_b5_check.json") -> int:
    import theirdir_judge as J
    import llm_api
    _key_env()
    items = items or sample()
    hk = J.judge_haiku_batch(items, poll_s=15)
    gm, usage = [], 0
    for s in range(0, len(items), 20):
        chunk = items[s:s + 20]
        raw, meta = llm_api.call_gemini(gemini_prompt(chunk), llm_api.read_key("gemini"), llm_api.MODELS["gemini"])
        u = meta.get("usage") or {}
        usage += (u.get("promptTokenCount") or 0) * 2 + ((u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)) * 12
        gm += parse_gemini_list(raw, len(chunk))
    k4 = cohen_kappa(hk, gm)
    collapse = lambda v: "CORRECT" if v == "CORRECT" else "NOT"
    k2 = cohen_kappa([collapse(v) for v in hk], [collapse(v) for v in gm])
    rep = dict(n=len(items), seed=SEED, kappa_4class=round(k4, 3), kappa_correct_vs_not=round(k2, 3), bar=L.KAPPA_BAR,
               passes_4class=k4 >= L.KAPPA_BAR, passes_correct_vs_not=k2 >= L.KAPPA_BAR,
               haiku=dict(Counter(hk)), gemini=dict(Counter(gm)), agree=round(sum(a == b for a, b in zip(hk, gm)) / len(hk), 3),
               gemini_cost_usd_std_price=round(usage / 1e6, 3), model_judges=[L.JUDGE_MODEL, llm_api.MODELS["gemini"]],
               episodes=[dict(src=i["src"], qid=i["qid"], ptype=i["ptype"], string=i["string_outcome"], haiku=h, gemini=g)
                         for i, h, g in zip(items, hk, gm)])
    (OUT / out_name).write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "episodes"})
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "live":
        sys.exit(live())
    if cmd == "sample":
        s = sample()
        print(len(s), Counter(r["string_outcome"] for r in s))
    elif cmd == "grade":
        sys.exit(grade())
    elif cmd in ("full-sample", "full-grade"):
        # prereg sec. 6: 1,500 episodes from baseline, real direction, 3 dummies. The 3 dummies (chosen 25.11.,
        # AFTER the run, said openly): the two shuffled dummies that decide B2 (shuf0, shuf8) + the first random one.
        s = sample(1500, OUT / "full", FULL_FILES)
        print(len(s), Counter(r["string_outcome"] for r in s), Counter(r["src"] for r in s))
        if cmd == "full-grade":
            sys.exit(grade(s, "kappa_b5_full.json"))
    else:
        print(__doc__)
