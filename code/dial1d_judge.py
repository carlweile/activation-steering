"""Dial 1d: BLIND judging = dial 1c's judge, rubrics, parser and Batch transport, own files, TWO phases and ONE cap guard.

    python dial1d_judge.py calibrate feedback|test   # 17 cases, needs >= 16 before any grade; adds 0.07 USD (single-call price) to the ledger
    python dial1d_judge.py pool feedback|test        # ONE shuffled blind pool + key per phase (never rebuilt)
    python dial1d_judge.py cost feedback|test        # projected USD for the ungraded part
    python dial1d_judge.py submit feedback|test N    # submit the first N ungraded chunks of 20 as one Batch job (refuses past the cap)
    python dial1d_judge.py poll feedback|test        # wait for the open job, write grades, print cost
Cap guard: running total = measured cost of finished jobs (tokens in the jobs file, standard price / 2) + calibrations;
price per answer = the larger of 1.92 / 1000 and the highest per-answer Batch price measured so far in this run. Before EVERY job the
whole ungraded rest of the phase is added (so before the first test job the WHOLE test pool): refuse if the sum passes CAP (15 USD).
"""
import json
import random
import re
import sys

import dial1b_judge as B
import dial1b_judge_batchapi as BA
import dial1c_judge as C
from dial1_lib import DATA, rows

RES, J = B.RES, B.J
SEED = 20261142
CAP = 15.0
USD_PER_ITEM = 1.92 / 1000
CAL_USD = 0.07
LEDGER = RES / "dial1d_spend.json"


def files(phase):
    p = "feedback" if phase == "feedback" else "test"
    return dict(comps=RES / ("dial1d_feedback_completions.jsonl" if p == "feedback" else "dial1d_completions.jsonl"),
                pool=RES / f"dial1d_{p}_pool.jsonl", key=RES / f"dial1d_{p}_pool_key.json",
                gr=RES / f"dial1d_{p}_grades_gemini.json", jobs=RES / f"dial1d_{p}_batch_jobs.json")


def items():
    it = {}
    for f, kind in (("dial1c_screen.jsonl", None), ("dial1c_gih.jsonl", None), ("dial1d_gi_t.jsonl", None), ("user_right.jsonl", "ur"), ("push50.jsonl", "push")):
        if (DATA / f).exists():
            for r in rows(f):
                it[r["id"]] = dict(r, kind=kind or r["kind"])
    return it


def aim(phase):
    f = files(phase)
    B.POOL, B.GR, B.items, BA.JOBS = f["pool"], f["gr"], items, f["jobs"]
    if not f["gr"].exists():
        f["gr"].write_text(json.dumps(dict(model=J.llm_api.MODELS["gemini"], grades={})))
    return f


def ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(calibrations=[])


def calibrate(phase):
    out = []
    for kind, ctx, ans, want, correct, wrong in C.CAL:
        raw, _ = J.llm_api.call_gemini(B.prompt_for(kind, ctx, ans, correct, wrong), J.llm_api.read_key("gemini"), J.llm_api.MODELS["gemini"])
        try:
            g, _ = B.parse(kind, json.loads(re.search(r"\{.*?\}", raw, re.S).group(0)))
        except Exception:
            g = "unparsed"
        out.append(dict(kind=kind, want=want, got=g, ok=g == want))
    n = sum(o["ok"] for o in out)
    (RES / f"dial1d_calibration_{phase}.json").write_text(json.dumps(dict(passed=n, of=len(C.CAL), line=16, model=J.llm_api.MODELS["gemini"], cases=out), indent=1))
    lg = ledger()
    lg["calibrations"].append(dict(phase=phase, usd=CAL_USD, passed=n))
    LEDGER.write_text(json.dumps(lg, indent=1))
    print("calibration", phase, n, "/", len(C.CAL), "(line 16)")
    if n < 16:
        raise SystemExit("calibration under 16 of 17 - stop, no grading")


def build_pool(phase):
    f = aim(phase)
    if f["pool"].exists():
        raise SystemExit("pool already exists - never rebuild a pool after grading started")
    rws = [json.loads(x) for x in open(f["comps"], encoding="utf-8") if x.strip()]
    random.Random(SEED).shuffle(rws)
    tag = phase[0]
    key = {f"{tag}{i:05d}": dict(arm=r["arm"], set=r["set"], id=r["id"]) for i, r in enumerate(rws)}
    pool = [dict(pid=f"{tag}{i:05d}", id=r["id"], answer=J.clean(r["completion"])) for i, r in enumerate(rws)]
    f["pool"].write_text("\n".join(json.dumps(p) for p in pool), encoding="utf-8")
    f["key"].write_text(json.dumps(dict(seed=SEED, key=key)))
    print("pool", len(pool))


def jobs_all():
    out = []
    for p in ("feedback", "test"):
        if files(p)["jobs"].exists():
            out += json.loads(files(p)["jobs"].read_text())
    return out


def price_per_item():
    """Highest measured per-answer Batch price in this run, but never below the 1c figure."""
    ms = [j["usd_standard_price"] / 2 / j["items"] for j in jobs_all() if j.get("usd_standard_price") and j.get("items")]
    return max([USD_PER_ITEM] + ms)


def spent():
    return sum(j.get("usd_standard_price", 0) for j in jobs_all()) / 2 + sum(c["usd"] for c in ledger()["calibrations"])


def guard(n_items, total_spent, price, cap=CAP):
    """Pure check, unit-tested: projected running total after this job's whole ungraded rest."""
    proj = total_spent + n_items * price
    if proj > cap:
        raise SystemExit(f"REFUSED: projected running total {proj:.2f} USD would pass the {cap} USD cap")
    return proj


def cost(phase):
    aim(phase)
    n = sum(map(len, BA.todo_chunks()))
    print(f"{phase}: {n} ungraded answers -> projected {n * price_per_item():.2f} USD at Batch price; spent so far {spent():.2f}")


def submit(phase, n_chunks):
    aim(phase)
    todo = BA.todo_chunks()
    proj = guard(sum(map(len, todo)), spent(), price_per_item())
    print(f"guard ok: projected running total {proj:.2f} USD (cap {CAP})")
    BA.submit(n_chunks)


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "calibrate":
        calibrate(a[0])
    elif c == "pool":
        build_pool(a[0])
    elif c == "cost":
        cost(a[0])
    elif c == "submit":
        submit(a[0], int(a[1]))
    elif c == "poll":
        aim(a[0])
        BA.poll()
