"""Post-hoc numbers for the 7B full run (looked at AFTER the run; they do not change the verdict).

1. cosine of each of the 10 shuffled dummies to the real direction (rebuilt exactly as the run built them:
   same stored activations g1_acts.pt, same seed 2026, same 0.3 leak filter);
2. the flip rate counted over ALL scored replies (flip / (flip + hold + abandon)) next to the prereg's
   flip rate (flip / (flip + hold)), per arm, as a rise over the baseline.

CPU only, no model, no API, 0 USD. Needs results/theirdir/full/g1_acts.pt (10 MB of activations, not in this
public copy: re-created by `theirdir_run.py full`) and their files (`theirdir_run.py fetch`). Writes results/theirdir/full_posthoc.json.
Run: .venv-tl\\Scripts\\python.exe theirdir_posthoc.py
"""
from __future__ import annotations

import json

import torch

import theirdir_lib as L
import theirdir_run as R

FULL = L.HERE / "results" / "theirdir" / "full"
LAYER = 19  # summary.json "layer"


def all_scored_flip(rows):
    rs = [r for r in rows if r["outcome"] is not None]
    return sum(r["outcome"] == "flip" for r in rs) / len(rs)


def main():
    real = R.load_dirs("capitulation")["dirs"][LAYER].float()
    real = real / real.norm()
    pairs = R.jl(R.UP / "outputs/qwen7b/activations/pairs_capitulation.jsonl")
    g1 = torch.load(FULL / "g1_acts.pt", weights_only=True)
    qids = [p["qid"] for p in pairs]
    shuf, skipped = L.shuffled_dummies(g1["acts"], g1["labels"], qids, real, n=10, seed=2026)
    cos = {f"shuf{i}": round(float(d @ real), 4) for i, d in enumerate(shuf)}

    base = R.jl(FULL / "g2_baseline.jsonl")
    fb, fb_all = L.flip_rate(base), all_scored_flip(base)
    arms = {}
    for p in sorted(FULL.glob("test_*_m8.jsonl")):
        rows = R.jl(p)
        name = p.stem[len("test_"):]
        arms[name] = {"flip_rise_prereg": round(L.flip_rate(rows) - fb, 4),
                      "flip_rise_all_scored": round(all_scored_flip(rows) - fb_all, 4),
                      "n": len(rows)}
    out = {"note": "post-hoc, chosen after the run; does not change the registered verdict",
           "baseline": {"flip_prereg": round(fb, 4), "flip_all_scored": round(fb_all, 4), "n": len(base)},
           "shuffle_skipped": skipped, "cos_to_real": cos, "arms_m8": arms}
    (L.HERE / "results" / "theirdir" / "full_posthoc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"cos": cos, "shuf0": arms.get("shuf0_m8"), "shuf8": arms.get("shuf8_m8"),
                      "real": arms.get("real_m8"), "baseline": out["baseline"]}, indent=1))


if __name__ == "__main__":
    main()
