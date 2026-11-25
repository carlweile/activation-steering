"""The full pipeline (prereg sec. 3): G1 -> G2 -> cost stop -> G3 -> G4 -> test -> H2.

Real run (RENTED pod only): python theirdir_run.py full --usd-per-h <price>
Dry run (LOCAL, free):      python theirdir_run.py full --dry --model Qwen/Qwen2.5-0.5B-Instruct --n 4
  --dry: small model, random stand-in directions, string judge, gates reported but never stop the flow.
Writes results/theirdir/<full|dry>/; every condition file is reused if present (resumable).
Verdict numbers that need the Gemini sample (B5 kappa) are filled by the report step after the run.
"""
from __future__ import annotations

import json

import torch

import theirdir_lib as L
import theirdir_run as R

LAYER_7B, MULTS = 19, (2, 4, 8)


def _stop(out, gate, why, dry):
    print(f"[{gate}] STOP: {why}")
    (out / "stops.jsonl").open("a", encoding="utf-8").write(json.dumps({"gate": gate, "why": why}) + "\n")
    return not dry  # True = really stop


def full(a) -> int:
    dry = getattr(a, "dry", False)
    model_name = a.model if dry else L.MODEL_7B
    out = L.HERE / "results" / "theirdir" / (getattr(a, "run_name", "") or ("dry" if dry else "full"))
    out.mkdir(parents=True, exist_ok=True)
    mode = "heuristic" if dry else "haiku"
    if not dry and not R.g0():
        return 1
    model = R.load_model(model_name, a.device)
    nl, dm = model.cfg.n_layers, model.cfg.d_model
    layer = nl * LAYER_7B // 28 if dry else LAYER_7B
    qs_all = R.jl(R.UP / "data/questions_qwen7b.jsonl")
    qmap = {q["id"]: q for q in qs_all}
    held = [q for q in qs_all if L.split_of(q["id"]) == "heldout"]
    extr = [q for q in qs_all if L.split_of(q["id"]) == "extract"]
    if dry:
        held, extr = held[:a.n], extr[:a.n]
    tpl, clock = L.load_templates(), R.Clock()
    fu = R.pushback_followups(tpl)
    hook = lambda d, al: [(f"blocks.{layer}.hook_resid_pre", L.make_steer_hook(d, al))]

    # ---- directions + G1 (re-cache their pairs at the layer; cosine to their stored direction)
    if dry:
        real, r1 = L.random_dummies(dm, 2, seed=11)
    else:
        real = R.load_dirs("capitulation")["dirs"][layer].float()
        r1 = R.load_dirs("pushback")["dirs"][layer].float()
    acts_p = out / "g1_acts.pt"
    pairs = R.jl(R.UP / "outputs/qwen7b/activations/pairs_capitulation.jsonl")
    if dry:
        pairs = pairs[:60]
    if acts_p.exists():
        g1 = torch.load(acts_p, weights_only=True)
    else:
        acts = torch.stack([R.resid_at(model, p["prompt"], layer) for p in pairs])
        g1 = {"acts": acts, "labels": torch.tensor([1 if p["label"] == "pos" else 0 for p in pairs])}
        torch.save(g1, acts_p)
    qids = [p["qid"] for p in pairs]
    rebuilt = L.matched_direction(g1["acts"], g1["labels"], qids)
    cos = float(rebuilt @ real) if rebuilt is not None else float("nan")
    print(f"[G1] cosine rebuilt vs theirs = {cos:.4f}")
    if not cos >= R.G1_COS and _stop(out, "G1", f"cosine {cos:.4f} < {R.G1_COS}", dry):
        return 1
    pos = g1["acts"][g1["labels"] == 1]
    alpha = {m: L.alpha_for(pos, real, m) for m in MULTS}

    # ---- batch guard (the author 20.11., plan A): batched only if byte-identical on THIS GPU, else slow path
    if getattr(a, "batch_size", 1) > 1:
        from theirdir_batchcheck import condition_prompts, guard
        (clock.batch_size, clock.equal_only), rep = guard(
            model, condition_prompts(model, held[:5], layer, real, alpha[MULTS[-1]]), a.batch_size)
        (out / "batch_guard.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"[batch] using batch_size={clock.batch_size} equal_only={clock.equal_only}")

    # ---- G2 baseline (TL, unhooked) on held-out
    init = R.turn1(model, held, clock)
    base = R.condition(out, "g2_baseline", lambda: R.turn2(model, held, init, fu, clock), qmap, mode)
    fb = L.flip_rate(base)
    theirs = {(r["qid"], r["pushback_type"]): r.get("final_verdict")
              for r in R.jl(R.UP / "outputs/qwen7b/transcripts/baseline_tl.jsonl")
              if r["pushback_type"] != "none" and r.get("final_verdict")}
    same = [r["final_verdict"] == theirs[(r["qid"], r["ptype"])] for r in base if (r["qid"], r["ptype"]) in theirs]
    agree = sum(same) / len(same) if same else float("nan")
    print(f"[G2] flip {fb:.3f} (theirs {R.THEIR_BASE_FLIP}), label agreement {agree:.3f} on {len(same)}")
    if not (abs(fb - R.THEIR_BASE_FLIP) <= R.G2_TOL and agree >= R.G2_AGREE) and \
            _stop(out, "G2", f"flip {fb:.3f}, agreement {agree:.3f}", dry):
        return 1

    # ---- cost stop (prereg sec. 3): project from measured speed
    proj = L.project_cost(clock.n, clock.t, R.TOTAL_REPLIES, a.usd_per_h, R.JUDGE_USD, 0.0)
    print(f"[cost] {clock.t / max(clock.n, 1):.2f} s/reply -> projected {proj:.2f} USD (cap {R.CAP_USD})")
    if L.over_cap(proj, R.CAP_USD) and _stop(out, "COST", f"projected {proj:.2f} USD > cap", dry):
        return 2

    # ---- G3 positive control: their ablation, every layer, both turns
    ab_h = L.ablation_hooks(nl, real)
    ab = R.condition(out, "g3_ablated",
                     lambda: R.turn2(model, held, R.turn1(model, held, clock, ab_h), fu, clock, ab_h), qmap, mode)
    d3, lo3, hi3 = L.bootstrap_diff(ab, base, seed=0)
    print(f"[G3] ablation flip change {d3:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
    if not hi3 < 0 and _stop(out, "G3", f"ablation CI upper {hi3:+.3f} not below 0", dry):
        return 1

    # ---- G4 strength pick on EXTRACTION questions only
    init_x = R.turn1(model, extr, clock)
    bx = R.condition(out, "g4_base", lambda: R.turn2(model, extr, init_x, fu, clock), qmap, mode)
    ab_by_m = {m: L.abandon_rate(R.condition(out, f"g4_m{m}",
                                             lambda m=m: R.turn2(model, extr, init_x, fu, clock, hook(real, alpha[m])),
                                             qmap, mode)) for m in MULTS}
    m = L.pick_m(ab_by_m, L.abandon_rate(bx))
    print(f"[G4] abandon by m {ab_by_m}, base {L.abandon_rate(bx):.3f} -> m = {m}")
    if m is None:
        if _stop(out, "G4", "no usable strength", dry):
            return 1
        m = MULTS[0]

    # ---- test on held-out at m: real, 10 shuffled, 5 random, R1; real at other m (descriptive)
    shuf, skipped = L.shuffled_dummies(g1["acts"], g1["labels"], qids, real, n=10, seed=2026)
    rand = L.random_dummies(dm, 5, seed=2027)
    arms = {"real": real, "r1_pushback": r1, **{f"shuf{i}": d for i, d in enumerate(shuf)},
            **{f"rand{i}": d for i, d in enumerate(rand)}}
    res = {name: R.condition(out, f"test_{name}_m{m}",
                             lambda d=d: R.turn2(model, held, init, fu, clock, hook(d, alpha[m])), qmap, mode)
           for name, d in arms.items()}
    for mm in MULTS:
        if mm != m:
            R.condition(out, f"test_real_m{mm}", lambda mm=mm: R.turn2(model, held, init, fu, clock,
                                                                        hook(real, alpha[mm])), qmap, mode)

    # ---- H2: neutral follow-up, no pushback
    neu = [("neutral", lambda q: L.NEUTRAL_FOLLOWUP)]
    h2 = {"base": R.condition(out, "h2_base", lambda: R.turn2(model, held, init, neu, clock), qmap, mode)}
    for name in ["real"] + [f"shuf{i}" for i in range(10)]:
        h2[name] = R.condition(out, f"h2_{name}_m{m}",
                               lambda d=arms[name]: R.turn2(model, held, init, neu, clock, hook(d, alpha[m])),
                               qmap, mode)

    # ---- numbers for the verdict (kappa filled by the report step after the Gemini sample)
    rise, lo, hi = L.bootstrap_diff(res["real"], base, seed=0)
    s_rows = lambda rows: [{"qid": r["qid"], "outcome": r["string_outcome"]} for r in rows]
    summary = {"model": model_name, "dry": dry, "layer": layer, "alpha": alpha, "m": m, "g1_cos": cos,
               "g2": {"flip": fb, "agree": agree}, "g3": [d3, lo3, hi3], "g4": ab_by_m, "shuffle_skipped": skipped,
               "rise": rise, "rise_ci": [lo, hi],
               "shuffled_rises": [L.flip_rate(res[f"shuf{i}"]) - fb for i in range(10)],
               "random_rises": [L.flip_rate(res[f"rand{i}"]) - fb for i in range(5)],
               "r1_rise": L.flip_rate(res["r1_pushback"]) - fb,
               "abandon_rise": L.abandon_rate(res["real"]) - L.abandon_rate(base),
               "string_rise": L.flip_rate(s_rows(res["real"])) - L.flip_rate(s_rows(base)),
               "h2": {k: L.flip_rate(v) for k, v in h2.items()},
               "replies": clock.n, "gpu_seconds": clock.t}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0
