"""Controls run, arm e (CPU): shuffled baseline for cos(ANS, AYS) = 0.68 at block 17.
Prereg: results/prereg_controls_2026-11-17.md, arm e + signed line a5 (2026-11-18). No model, no GPU, no judge.

Null 1 (sign flips): each pair difference gets a random sign before averaging, separately for ANS and AYS; cos of the two.
Null 2 (template-label permutation, the Lee-style null): pool the ANS and AYS pairs of one training fold, split them at
random into two groups of the real sizes, direction of each = mean diff, cos of the two. Keeps the shared
"user pushes back" cue. 1,000 draws per fold, fixed seeds.
Pass (signed): real cos above the 95th percentile of null 2. Main number (signed): gap = real cos - median of null 2.

    python controls_arm_e.py  (CUDA torch)   -> results/controls_arm_e.json
"""
import json

import torch

import dial1_lib as L
import dial1_percell_lib as P

R = L.HERE / "results"
N_DRAWS = 1000
SEED = 20261118


def cos(a, b):
    return torch.dot(a, b).item()


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def fold_null(diffs, ia, iy, fold_seed):
    real = cos(P.direction(diffs, ia), P.direction(diffs, iy))
    pool = torch.tensor(ia + iy)
    g = torch.Generator().manual_seed(fold_seed)
    null1, null2 = [], []
    for _ in range(N_DRAWS):
        perm = pool[torch.randperm(len(pool), generator=g)].tolist()
        null2.append(cos(P.direction(diffs, perm[:len(ia)]), P.direction(diffs, perm[len(ia):])))
        sa = (torch.randint(0, 2, (len(ia),), generator=g) * 2 - 1).float().unsqueeze(1)
        sy = (torch.randint(0, 2, (len(iy),), generator=g) * 2 - 1).float().unsqueeze(1)
        null1.append(cos(P.unit((diffs[ia] * sa).mean(0)), P.unit((diffs[iy] * sy).mean(0))))
    p95 = pct(null2, 0.95)
    med2 = pct(null2, 0.5)
    return dict(n_ans=len(ia), n_ays=len(iy), real=round(real, 4),
                null1=dict(median=round(pct(null1, 0.5), 4), p95=round(pct(null1, 0.95), 4), max=round(max(null1), 4),
                           share_ge_real=sum(x >= real for x in null1) / N_DRAWS),
                null2=dict(median=round(med2, 4), p05=round(pct(null2, 0.05), 4), p95=round(p95, 4),
                           max=round(max(null2), 4), share_ge_real=sum(x >= real for x in null2) / N_DRAWS),
                gap_real_minus_null2_median=round(real - med2, 4),
                above_p95_null2=real > p95)


def main():
    ps = P.pairs()
    fold = json.loads((R / "dial1_percell_folds.json").read_text())["fold"]
    saved = torch.load(R / "dial1_rebuild_pairs.pt", weights_only=False)
    assert saved["ids"] == [p["id"] for p in ps], "diff rows are not in pair order"
    diffs = saved["diffs"].float()
    out = {}
    for k, tf in enumerate("AB"):
        ia = P.idx_of(ps, fold, ("answer",), tf)
        iy = P.idx_of(ps, fold, ("areyousure",), tf)
        out[tf] = fold_null(diffs, ia, iy, SEED + k)
    stored = json.loads((R / "crossmap_dirs.json").read_text())["cosines"]["within_fold"]
    for tf in "AB":
        assert abs(out[tf]["real"] - stored[tf]["ANS|AYS"]) < 1e-3, f"real cos {tf} != stored crossmap value"
    passed = all(out[tf]["above_p95_null2"] for tf in "AB")
    doc = dict(prereg="results/prereg_controls_2026-11-17.md (arm e, signed a5 2026-11-18)", layer=L.LAYER,
               n_draws=N_DRAWS, seed=SEED, folds=out,
               verdict="PASS" if passed else "FAIL",
               rule="real cos above the 95th percentile of null 2 in BOTH folds; main number = gap to null-2 median")
    (R / "controls_arm_e.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
