"""Dial 1b: the GPU run (prereg section 3): gates, 13 arms on GI-200, baseline + real on UR-40 / PUSH-50, HELD-AB.

    python dial1b_sweep.py        # ~20 min on the 4070; writes results/dial1b_completions.jsonl + dial1b_sweep.json
"""
import datetime
import json

import torch

import dial1_lib as L

RES = L.HERE / "results"
F_MAIN, F_HI = 0.20, 0.35


def main():
    tok, model = L.load()
    cap = {}
    h = model.model.layers[L.LAYER].register_forward_hook(lambda m, i, o: cap.update(x=(o[0] if isinstance(o, tuple) else o)))
    with torch.no_grad():
        enc = tok([L.chat(tok, "Hello there")], return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[L.LAYER + 1]
    idx_diff = (cap["x"] - hs).abs().max().item()
    h.remove()
    if idx_diff != 0:
        raise SystemExit(f"index check failed ({idx_diff}) - stop")
    dial = L.Dial(model)
    st = torch.load(RES / "dial1b_dirs.pt")
    d, ctrl, norm = st["gi_dir"].float(), st["controls"].float(), st["norm"]
    gi, ur, push, held = L.rows("gi200.jsonl"), L.rows("user_right.jsonl"), L.rows("push50.jsonl"), L.rows("held_ab.jsonl")
    old_ur = {}
    for l in open(RES / "dial1_dom_completions.jsonl", encoding="utf-8"):
        c = json.loads(l)
        if c["set"] == "user_right" and c["arm"] == "baseline_1":
            old_ur[c["id"]] = c["completion"]
    held_stored = json.loads((RES / "dial1_dom_phaseA.json").read_text())["arms"]["baseline_1"]["held_ab_psyc"]
    comps, arms = [], {}

    def run(name, mode, sets, with_held):
        dial.set(mode)
        rec = dict(alpha=None if mode is None else mode[2])
        if with_held:
            rec["held_ab_psyc"] = sum(L.p_syc(tok, model, held)) / len(held)
        for sname, items in sets:
            outs = L.generate(tok, model, [it["prompt"] for it in items], 128)
            comps.extend(dict(arm=name, set=sname, id=it["id"], prompt=it["prompt"], completion=o) for it, o in zip(items, outs))
            rec[f"{sname}_mean_words"] = sum(len(o.split()) for o in outs) / len(outs)
        dial.set(None)
        arms[name] = rec
        print(name, rec, flush=True)

    full = [("gi200", gi), ("user_right", ur), ("push50", push)]
    run("baseline", None, full, True)
    n_diff = sum(c["completion"] != old_ur[c["id"]] for c in comps if c["set"] == "user_right")
    gate = dict(index_check=idx_diff, ur_diff_vs_run0b=n_diff, held_baseline=arms["baseline"]["held_ab_psyc"], held_stored=held_stored)
    print("gates", gate, flush=True)
    if n_diff > 2 or abs(gate["held_baseline"] - held_stored) > 0.002:
        (RES / "dial1b_sweep.json").write_text(json.dumps(dict(gate=gate, stopped=True), indent=1))
        raise SystemExit("baseline gate failed - stop, no steered arms")
    run("real_f0.20", ("add", d, F_MAIN * norm), full, True)
    run("real_f0.35", ("add", d, F_HI * norm), [("gi200", gi)], True)
    for k in range(10):
        run(f"ctrl{k}_f0.20", ("add", ctrl[k], F_MAIN * norm), [("gi200", gi)], False)
    (RES / "dial1b_completions.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in comps), encoding="utf-8")
    (RES / "dial1b_sweep.json").write_text(json.dumps(dict(date=str(datetime.date.today()), model=L.MODEL_ID, layer=L.LAYER,
                                                         norm=norm, gate=gate, arms=arms, n_completions=len(comps)), indent=1))


if __name__ == "__main__":
    main()
