"""Dial 1d sitting 2: all arms of prereg 3.11 on GI-T, UR-40 and PUSH-50, plus HELD-AB.

    python dial1d_arms.py         # GPU ~55 min; resumable (each arm is saved when it finishes)

Full run (20 arms on GI-T): baseline; a1_f0.20; single_f0.30 (a1 at f 0.30); primary (a1 @17 f 0.20 + a2 @18 f 0.10); dGI_f0.30 (the cue-built
1c arrow); pair_k (c1_k @17 + c2_k @18, k = 0..9); real_a1_c2_k (a1 + c2_k, k = 0..4). Branch 3.5-6-iii (stop iii recorded): baseline, a1_f0.20
(= primary), dGI_f0.30 and c1only_k (c1_k alone @17 f 0.20, k = 0..9). UR-40 + PUSH-50 for baseline and the primary arm; HELD-AB the same.
Vectors are LOADED from results/dial1d_arrow*.pt, never rebuilt.
"""
import datetime
import json
import time

import torch

import dial1_lib as L
import dial1d_lib as DL
import dial1d_run as R
import dial1d_sets as D

RES = L.HERE / "results"
ARMS = RES / "dial1d_arms"
F1, F2, F_SINGLE = R.F1, R.F2, R.F_SINGLE


def arm_list(branch3, a1, a2, c1, c2, gi_dir):
    """[(name, m17, m18, full)]  full = also UR-40 / PUSH-50 / HELD-AB. Modes are (unit arrow, alpha) or None."""
    m17 = lambda v, f: (v, DL.alpha17(f))
    m18 = lambda v, f: (v, DL.alpha18(f))
    arms = [("baseline", None, None, True)]
    if branch3:
        arms += [("a1_f0.20", m17(a1, F1), None, True), ("dGI_f0.30", m17(gi_dir, F_SINGLE), None, False)]
        arms += [(f"c1only_{k}", m17(c1[k], F1), None, False) for k in range(10)]
        return arms
    arms += [("a1_f0.20", m17(a1, F1), None, False), ("single_f0.30", m17(a1, F_SINGLE), None, False),
             ("primary", m17(a1, F1), m18(a2, F2), True), ("dGI_f0.30", m17(gi_dir, F_SINGLE), None, False)]
    arms += [(f"pair_{k}", m17(c1[k], F1), m18(c2[k], F2), False) for k in range(10)]
    arms += [(f"real_a1_c2_{k}", m17(a1, F1), m18(c2[k], F2), False) for k in range(5)]
    return arms


def main():
    ARMS.mkdir(exist_ok=True)
    stp = json.loads((RES / "dial1d_stop.json").read_text()) if (RES / "dial1d_stop.json").exists() else {}
    if stp and stp.get("stop") != "iii":
        raise SystemExit(f"stop recorded ({stp['stop']}) - no arms")
    branch3 = stp.get("stop") == "iii"
    sets = json.loads((RES / "dial1d_sets.json").read_text())
    gt = L.rows("dial1d_gi_t.jsonl")
    build, feedback = D.load_roles()
    unused, _ = D.stored_unused()
    D.assert_roles_disjoint(build, feedback, gt, sets["n"], unused, D.kept_new() if sets["n"] == 300 else None)
    a1c = torch.load(RES / "dial1d_arrow1.pt")
    a2c = torch.load(RES / "dial1d_arrow2.pt") if not branch3 else {}
    gi_dir = torch.load(RES / "dial1b_dirs.pt")["gi_dir"].float()
    tok, model, dial = R.load_model()
    ur, push, held = L.rows("user_right.jsonl"), L.rows("push50.jsonl"), L.rows("held_ab.jsonl")
    old_ur = {c["id"]: c["completion"] for c in R.jl(RES / "dial1_dom_completions.jsonl") if c["set"] == "user_right" and c["arm"] == "baseline_1"}
    held_stored = json.loads((RES / "dial1_dom_phaseA.json").read_text())["arms"]["baseline_1"]["held_ab_psyc"]

    def run(name, m17, m18, full):
        f = ARMS / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        t0 = time.time()
        dial.set(m17, m18)
        rec = dict(alpha17=None if m17 is None else m17[1], alpha18=None if m18 is None else m18[1], comps=[])
        sets_ = [("git", gt)] + ([("user_right", ur), ("push50", push)] if full else [])
        if full:
            rec["held_ab_psyc"] = sum(L.p_syc(tok, model, held)) / len(held)
        for sname, items in sets_:
            outs = L.generate(tok, model, [it["prompt"] for it in items], 128)
            rec["comps"] += [dict(arm=name, set=sname, id=it["id"], prompt=it["prompt"], completion=o) for it, o in zip(items, outs)]
            rec[f"{sname}_mean_words"] = sum(len(o.split()) for o in outs) / len(outs)
        dial.set()
        rec["seconds"] = round(time.time() - t0)
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(name, {k: v for k, v in rec.items() if k != "comps"}, flush=True)
        return rec

    lst = arm_list(branch3, a1c["a1"], a2c.get("a2"), a1c["c1"], a2c.get("c2"), gi_dir)
    base = run(*lst[0])
    n_diff = sum(c["completion"] != old_ur[c["id"]] for c in base["comps"] if c["set"] == "user_right")
    gate = dict(ur40_equal=40 - n_diff, held_baseline=base["held_ab_psyc"], held_stored=held_stored)
    print("gates", gate, flush=True)
    if 40 - n_diff < R.GATE_UR or abs(gate["held_baseline"] - held_stored) > R.HELD_TOL:
        R.stop("baseline moved", "arms-sitting baseline gate failed (R6: question for the author)", gates=gate)
        raise SystemExit(1)
    for a in lst[1:]:
        run(*a)
    comps, arms = [], {}
    for f in sorted(ARMS.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        comps += rec.pop("comps")
        arms[f.stem] = rec
    (RES / "dial1d_completions.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in comps), encoding="utf-8")
    (RES / "dial1d_sweep.json").write_text(json.dumps(dict(date=str(datetime.date.today()), model=L.MODEL_ID, branch_iii=branch3, gate=gate, arms=arms,
                                                         n_completions=len(comps), gpu_seconds=sum(a["seconds"] for a in arms.values())), indent=1))
    print("done", len(comps), "completions")


if __name__ == "__main__":
    main()
