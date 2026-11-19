"""Controls run (prereg results/prereg_controls_2026-11-17.md): BLIND judging = dial 1c / 1d's judge, rubrics, parser and Batch
transport; own files per arm; ONE cap guard PER CARD (signed a6: 3 cards, each under 15 USD at Batch price).

    python controls_judge.py calibrate <arm>       # 17 cases, needs >= 16 before any grade; 0.07 USD into the card ledger
    python controls_judge.py pool <arm>            # ONE shuffled blind pool + key (never rebuilt)
    python controls_judge.py cost <arm>
    python controls_judge.py submit <arm> N        # first N ungraded chunks of 20 as one Batch job (refuses past the card cap)
    python controls_judge.py poll <arm>
<arm> = b | a | c | c2 | f. Card of each arm: b -> card1, a -> card2, c (plain band screen) + c2 (steered) -> card3,
f (-a1, signed 2026-11-18 evening) -> card4.
"""
import json
import random
import re
import sys

import dial1b_judge as B
import dial1b_judge_batchapi as BA
import dial1c_judge as C
import dial1d_judge as D1

RES, J = B.RES, B.J
CARD = {"b": "card1", "a": "card2", "c": "card3", "c2": "card3", "f": "card4"}
SEED = {"b": 20261181, "a": 20261182, "c": 20261183, "c2": 20261184, "f": 20261185}
CAP = 15.0


def files(arm):
    return dict(comps=RES / f"controls_arm_{arm}_completions.jsonl", pool=RES / f"controls_arm_{arm}_pool.jsonl",
                key=RES / f"controls_arm_{arm}_pool_key.json", gr=RES / f"controls_arm_{arm}_grades_gemini.json",
                jobs=RES / f"controls_arm_{arm}_batch_jobs.json")


def ledger_path(arm):
    return RES / f"controls_spend_{CARD[arm]}.json"


def items(arm):
    it = D1.items()
    extra = RES / f"controls_arm_{arm}_items.jsonl"   # arms with their own prompts (a: P1 / P2 variants, c: wrong-answer items)
    if extra.exists():
        for l in open(extra, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                it[r["id"]] = r
    return it


def aim(arm):
    f = files(arm)
    B.POOL, B.GR, BA.JOBS = f["pool"], f["gr"], f["jobs"]
    B.items = lambda: items(arm)
    if not f["gr"].exists():
        f["gr"].write_text(json.dumps(dict(model=J.llm_api.MODELS["gemini"], grades={})))
    return f


def ledger(arm):
    p = ledger_path(arm)
    return json.loads(p.read_text()) if p.exists() else dict(calibrations=[])


def calibrate(arm):
    out = []
    for kind, ctx, ans, want, correct, wrong in C.CAL:
        raw, _ = J.llm_api.call_gemini(B.prompt_for(kind, ctx, ans, correct, wrong), J.llm_api.read_key("gemini"), J.llm_api.MODELS["gemini"])
        try:
            g, _ = B.parse(kind, json.loads(re.search(r"\{.*?\}", raw, re.S).group(0)))
        except Exception:
            g = "unparsed"
        out.append(dict(kind=kind, want=want, got=g, ok=g == want))
    n = sum(o["ok"] for o in out)
    (RES / f"controls_calibration_{arm}.json").write_text(json.dumps(dict(passed=n, of=len(C.CAL), line=16, cases=out), indent=1))
    lg = ledger(arm)
    lg["calibrations"].append(dict(arm=arm, usd=D1.CAL_USD, passed=n))
    ledger_path(arm).write_text(json.dumps(lg, indent=1))
    print("calibration", arm, n, "/", len(C.CAL), "(line 16)")
    if n < 16:
        raise SystemExit("calibration under 16 of 17 - stop, no grading")


def build_pool(arm, only_arms=None):
    f = aim(arm)
    if f["pool"].exists():
        raise SystemExit("pool already exists - never rebuild a pool after grading started")
    rws = [json.loads(x) for x in open(f["comps"], encoding="utf-8") if x.strip()]
    if only_arms:
        rws = [r for r in rws if r["arm"] in only_arms]
    random.Random(SEED[arm]).shuffle(rws)
    key = {f"{arm}{i:05d}": dict(arm=r["arm"], set=r["set"], id=r["id"]) for i, r in enumerate(rws)}
    pool = [dict(pid=f"{arm}{i:05d}", id=r["id"], answer=J.clean(r["completion"])) for i, r in enumerate(rws)]
    f["pool"].write_text("\n".join(json.dumps(p) for p in pool), encoding="utf-8")
    f["key"].write_text(json.dumps(dict(seed=SEED[arm], key=key)))
    print("pool", len(pool))


def card_jobs(arm):
    out = []
    for a, c in CARD.items():
        if c == CARD[arm] and files(a)["jobs"].exists():
            out += json.loads(files(a)["jobs"].read_text())
    return out


def price_per_item(arm):
    ms = [j["usd_standard_price"] / 2 / j["items"] for j in card_jobs(arm) if j.get("usd_standard_price") and j.get("items")]
    return max([D1.USD_PER_ITEM] + ms)


def spent(arm):
    return sum(j.get("usd_standard_price", 0) for j in card_jobs(arm)) / 2 + sum(c["usd"] for c in ledger(arm)["calibrations"])


def cost(arm):
    aim(arm)
    n = sum(map(len, BA.todo_chunks()))
    print(f"arm {arm}: {n} ungraded -> projected {n * price_per_item(arm):.2f} USD Batch; card spent so far {spent(arm):.2f} (cap {CAP})")


def submit(arm, n_chunks):
    aim(arm)
    proj = D1.guard(sum(map(len, BA.todo_chunks())), spent(arm), price_per_item(arm), CAP)
    print(f"guard ok: projected card total {proj:.2f} USD (cap {CAP})")
    BA.submit(n_chunks)


def grades(arm):
    """{(arm_name, set, id): (grade, coherence)} from the pool key + grades file."""
    f = files(arm)
    key = json.loads(f["key"].read_text())["key"]
    gr = json.loads(f["gr"].read_text())["grades"]
    return {(key[p]["arm"], key[p]["set"], key[p]["id"]): (v["grade"], v["coherence"]) for p, v in gr.items()}


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
