"""Dial 1c: the GPU run (prereg section 5): gates, then 13 arms on GI-H, baseline + real-17 + real-18 on UR-40 / PUSH-50, HELD-AB.

    python dial1c_sweep.py       # ~35 min on the 4070; resumable (each arm is saved when it finishes)

Writes results/dial1c_arms/<arm>.json, then results/dial1c_completions.jsonl + dial1c_sweep.json. Vectors are LOADED, never rebuilt.
"""
import datetime
import json

import torch

import dial1_lib as L
from dial1c_gen import index_check

RES = L.HERE / "results"
ARMS = RES / "dial1c_arms"
F_MAIN = 0.20


def main():
    ARMS.mkdir(exist_ok=True)
    tok, model = L.load()
    index_check(tok, model)
    idx_diff = 0.0
    dial17 = L.Dial(model)
    L.LAYER = 18
    dial18 = L.Dial(model)
    L.LAYER = 17
    st = torch.load(RES / "dial1b_dirs.pt")
    d, ctrl, norm = st["gi_dir"].float(), st["controls"].float(), st["norm"]
    b18 = json.loads((RES / "dial1c_block18.json").read_text())
    use18 = b18["gate"] == "PASS"
    if use18:
        s18 = torch.load(RES / "dial1c_dir18.pt")
        d18, a18 = s18["dir18"].float(), s18["alpha18"]
    gih, ur, push, held = L.rows("dial1c_gih.jsonl"), L.rows("user_right.jsonl"), L.rows("push50.jsonl"), L.rows("held_ab.jsonl")
    old_ur = {}
    for l in open(RES / "dial1_dom_completions.jsonl", encoding="utf-8"):
        c = json.loads(l)
        if c["set"] == "user_right" and c["arm"] == "baseline_1":
            old_ur[c["id"]] = c["completion"]
    held_stored = json.loads((RES / "dial1_dom_phaseA.json").read_text())["arms"]["baseline_1"]["held_ab_psyc"]

    def run(name, dial, mode, sets, with_held):
        f = ARMS / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        dial.set(mode)
        rec = dict(alpha=None if mode is None else mode[2], comps=[])
        if with_held:
            rec["held_ab_psyc"] = sum(L.p_syc(tok, model, held)) / len(held)
        for sname, items in sets:
            outs = L.generate(tok, model, [it["prompt"] for it in items], 128)
            rec["comps"] += [dict(arm=name, set=sname, id=it["id"], prompt=it["prompt"], completion=o) for it, o in zip(items, outs)]
            rec[f"{sname}_mean_words"] = sum(len(o.split()) for o in outs) / len(outs)
        dial.set(None)
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(name, {k: v for k, v in rec.items() if k != "comps"}, flush=True)
        return rec

    full = [("gih", gih), ("user_right", ur), ("push50", push)]
    base = run("baseline", dial17, None, full, True)
    n_diff = sum(c["completion"] != old_ur[c["id"]] for c in base["comps"] if c["set"] == "user_right")
    gate = dict(index_check=idx_diff, ur_diff_vs_run0b=n_diff, held_baseline=base["held_ab_psyc"], held_stored=held_stored)
    print("gates", gate, flush=True)
    if n_diff > 2 or abs(gate["held_baseline"] - held_stored) > 0.002:
        (RES / "dial1c_sweep.json").write_text(json.dumps(dict(gate=gate, stopped=True), indent=1))
        raise SystemExit("baseline gate failed - stop, no steered arms")
    names = ["baseline", "real17_f0.20"]
    run("real17_f0.20", dial17, ("add", d, F_MAIN * norm), full + [], True)
    if use18:
        run("real18", dial18, ("add", d18, a18), full, False)
        names.append("real18")
    for k in range(10):
        run(f"ctrl{k}_f0.20", dial17, ("add", ctrl[k], F_MAIN * norm), [("gih", gih)], False)
        names.append(f"ctrl{k}_f0.20")
    arms, comps = {}, []
    for f in sorted(ARMS.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        comps += rec.pop("comps")
        arms[f.stem] = rec
    (RES / "dial1c_completions.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in comps), encoding="utf-8")
    (RES / "dial1c_sweep.json").write_text(json.dumps(dict(date=str(datetime.date.today()), model=L.MODEL_ID, layer=L.LAYER, norm=norm,
                                                         block18_used=use18, gate=gate, arms=arms, n_completions=len(comps)), indent=1))
    print("done", len(comps), "completions")


if __name__ == "__main__":
    main()
