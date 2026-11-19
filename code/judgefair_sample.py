"""Judge-fairness check: build the 600-reply frame, the 156-reply blind reading sheet and its private key.

    python judgefair_sample.py frame     # print group sizes only (no draw)
    python judgefair_sample.py cells     # frame sizes per stratum -> results/judgefair_frame_cells.json (for the scorer)
    python judgefair_sample.py draw      # the draw (fixed seed) -> results/judgefair_sample.json + results/judgefair_sheet/

Run under the lab's torch interpreter ONLY because dial1b_judge imports dial1_lib (torch) at import time: import, no model, no CUDA.
Inputs are stored files only. The sheet text is dial1b_judge.batch_text: the exact rubric, 700-char cut and chunks of 20 the Gemini judge got.
The judge label never reaches the sheet; it sits in the sample file for scoring only.
"""
import hashlib
import json
import random
import sys

import dial1b_judge as B
import dial1d_judge as D
from dial1b_report import present

RES = B.RES
SEED = 20261143
SHEET = RES / "judgefair_sheet"
OUT = RES / "judgefair_sample.json"
QUOTA = {1: 6, 2: 12, 3: 12}   # per arm, Part B (card default 60 = (6+12+12) * 2 arms)
LABELS = ("sycophantic", "independent")


def jl(name):
    return [json.loads(x) for x in open(RES / name, encoding="utf-8") if x.strip()]


def group(coh, has_correct, has_wrong):
    """G3 = judge-non-fluent; G1 = fluent and exactly one answer present; G2 = fluent and both or neither (mixed)."""
    if coh < 3:
        return 3
    return 1 if has_correct != has_wrong else 2


def frame():
    """600 rows: the 300 steered replies and the plain model's 300 replies to the same GI-H prompts, with the judge's grade."""
    gih = {r["id"]: r for r in B.rows("dial1c_gih.jsonl")}
    rows = []
    for arm, pool, key, gr in (("steered", "dial1d_feedback_pool.jsonl", "dial1d_feedback_pool_key.json", "dial1d_feedback_grades_gemini.json"),
                               ("baseline", "dial1c_judge_pool.jsonl", "dial1c_judge_pool_key.json", "dial1c_grades_gemini.json")):
        k = json.loads((RES / key).read_text())["key"]
        g = json.loads((RES / gr).read_text())["grades"]
        for p in jl(pool):
            kk = k[p["pid"]]
            if arm == "baseline" and not (kk["arm"] == "baseline" and kk["set"] == "gih"):
                continue
            it = gih[p["id"]]
            v = g[p["pid"]]
            c, w = present(p["answer"], it["correct"]), present(p["answer"], it["wrong"])
            rows.append(dict(arm=arm, id=p["id"], pid=p["pid"], answer=p["answer"], judge=v["grade"], coh=v["coherence"],
                             has_correct=c, has_wrong=w, group=group(v["coherence"], c, w)))
    return rows


def stop_set():
    """The 96 prompts the plain model caved on (stop (ii)'s base): the steered replies of exactly these ids."""
    k = json.loads((RES / "dial1c_judge_pool_key.json").read_text())["key"]
    g = json.loads((RES / "dial1c_grades_gemini.json").read_text())["grades"]
    caved = {k[p]["id"] for p, v in g.items() if k[p]["arm"] == "baseline" and k[p]["set"] == "gih" and v["grade"] == "sycophantic"}
    return caved


def pick_cell(cands, n, rnd):
    """n replies from one arm x group cell: every dodge first, then the rest split evenly sycophantic / independent."""
    cands = sorted(cands, key=lambda r: r["pid"])
    if len(cands) <= n:
        return cands
    out = [r for r in cands if r["judge"] == "dodge"][:n]
    rest = n - len(out)
    pools = {lab: [r for r in cands if r["judge"] == lab] for lab in LABELS}
    for r in pools.values():
        rnd.shuffle(r)
    want = {LABELS[0]: rest // 2 + rest % 2, LABELS[1]: rest // 2}
    take = {lab: min(want[lab], len(pools[lab])) for lab in LABELS}
    short = rest - sum(take.values())
    for lab in LABELS:                       # a short label gives its seats to the other
        extra = min(short, len(pools[lab]) - take[lab])
        take[lab] += extra
        short -= extra
    for lab in LABELS:
        out += pools[lab][:take[lab]]
    return out


def draw(rows):
    caved = stop_set()
    part_a = [dict(r, part="A") for r in rows if r["arm"] == "steered" and r["id"] in caved]
    taken = {r["pid"] for r in part_a}
    rest = [r for r in rows if r["pid"] not in taken]
    rnd = random.Random(SEED)
    part_b = []
    for arm in ("steered", "baseline"):
        for gno, n in QUOTA.items():
            part_b += [dict(r, part="B") for r in pick_cell([r for r in rest if r["arm"] == arm and r["group"] == gno], n, rnd)]
    return part_a + part_b


def write_sheet(sample):
    """Shuffle, number, chunk by 20, write the exact judge text per chunk. Key (opaque number -> pid) stays in the sample file."""
    rnd = random.Random(SEED + 1)
    order = list(sample)
    rnd.shuffle(order)
    it = D.items()
    SHEET.mkdir(exist_ok=True)
    chunks = [order[i:i + 20] for i in range(0, len(order), 20)]
    for n, ch in enumerate(chunks, 1):
        text = B.batch_text(it, [dict(id=r["id"], answer=r["answer"]) for r in ch])
        (SHEET / f"sheet_{n:02d}.txt").write_text(text, encoding="utf-8", newline="\n")
        for k, r in enumerate(ch, 1):
            r["sheet"], r["pos"] = n, k
    return order, len(chunks)


def main():
    rows = frame()
    sizes = {a: {g: sum(r["arm"] == a and r["group"] == g for r in rows) for g in (1, 2, 3)} for a in ("steered", "baseline")}
    print("frame", len(rows), sizes)
    if sys.argv[1] == "frame":
        return
    if sys.argv[1] == "cells":   # frame sizes per (arm, group, in-Part-A, judge label): lets the scorer weight with stdlib only
        caved = stop_set()
        cells = {}
        for r in rows:
            k = f"{r['arm']}|{r['group']}|{'A' if r['arm'] == 'steered' and r['id'] in caved else 'rest'}|{r['judge']}"
            cells[k] = cells.get(k, 0) + 1
        (RES / "judgefair_frame_cells.json").write_text(json.dumps(cells, indent=1, sort_keys=True))
        print("cells", len(cells))
        return
    sample = draw(rows)
    order, n = write_sheet(sample)
    ids = sorted(f"{r['arm']}:{r['pid']}" for r in sample)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    OUT.write_text(json.dumps(dict(seed=SEED, n=len(order), sheets=n, ids_sha256=digest,
                                   rows=[{k: v for k, v in r.items() if k != "answer"} for r in order]), indent=1))
    print("sample", len(order), "sheets", n, "sha256", digest)
    for part in "AB":
        print(part, {a: {g: sum(r["part"] == part and r["arm"] == a and r["group"] == g for r in order) for g in (1, 2, 3)} for a in ("steered", "baseline")})


if __name__ == "__main__":
    main()
