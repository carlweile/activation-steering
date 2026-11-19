"""Dial 1d: the stratified own-answer arrow builder, label shuffles, stability gate, S / H grouping and the stage stops.

Prereg results/prereg_dial1d_2026-11-11.md 3.2, 3.5, 3.6 and the run plan (readings R1-R4, R7-R8). CPU only, no model, no API.
Labels are "c" (caved), "h" (held, fluent) or None (left out: dodge, non-fluent, ungraded). X = one state vector per item.
"""
import random
import statistics

import torch

MIN_N = 4                    # a stratum counts only with >= 4 caved AND >= 4 held
GATE_COS = 0.50              # stability gate (prereg 3.5-1)
MIN_GROUP = 24               # smallest group S / H (prereg 3.5-3)
SEED_C1, SEED_C2, SEED_SWAP0 = 20261112, 20261122, 20261132
MAX_SWAPS, SWAP_COS = 5, 0.5


def unit(x):
    return x / x.norm()


def cos(a, b):
    return torch.dot(unit(a), unit(b)).item()


def strata_counts(y, st):
    out = {}
    for lab, s in zip(y, st):
        if lab in ("c", "h"):
            out.setdefault(s, dict(c=0, h=0))[lab] += 1
    return out


def usable_strata(y, st, min_n=MIN_N):
    return sorted(s for s, v in strata_counts(y, st).items() if v["c"] >= min_n and v["h"] >= min_n)


def build(X, y, st, strata=None, min_n=MIN_N):
    """Weighted sum of per-stratum (mean caved - mean held), weight nc*nh/(nc+nh), unit-normed. Returns (arrow, used strata)."""
    strata = usable_strata(y, st, min_n) if strata is None else strata
    acc, used = torch.zeros(X.shape[1]), []
    for s in strata:
        c = [i for i, (l, t) in enumerate(zip(y, st)) if l == "c" and t == s]
        h = [i for i, (l, t) in enumerate(zip(y, st)) if l == "h" and t == s]
        if not c or not h:
            continue
        acc += len(c) * len(h) / (len(c) + len(h)) * (X[c].mean(0) - X[h].mean(0))
        used.append(s)
    if not used:
        raise ValueError("no usable stratum")
    return unit(acc), used


def shuffle_labels(y, st, seed):
    """Permute the c / h labels inside each stratum (group sizes and strata kept; None stays None)."""
    rng, out = random.Random(seed), list(y)
    for s in sorted({t for l, t in zip(y, st) if l in ("c", "h")}, key=str):
        idx = [i for i, (l, t) in enumerate(zip(y, st)) if l in ("c", "h") and t == s]
        labs = [y[i] for i in idx]
        rng.shuffle(labs)
        for i, l in zip(idx, labs):
            out[i] = l
    return out


def make_controls(X, y, st, real, seed0, log, n=10):
    """Ten shuffled arrows (seeds seed0 + k). A control with |cos| >= 0.5 to the real arrow is replaced by the next spare seed
    (SEED_SWAP0, +1, ...), at most MAX_SWAPS times, each swap logged. Returns [(seed, arrow, labels)]."""
    spare, ctrls = SEED_SWAP0, []
    for k in range(n):
        seed = seed0 + k
        while True:
            lab = shuffle_labels(y, st, seed)
            try:
                a, _ = build(X, lab, st, usable_strata(y, st))
            except ValueError:
                a = None
            if a is not None and abs(cos(a, real)) < SWAP_COS:
                break
            log.append(dict(replaced_seed=seed, cos=None if a is None else cos(a, real)))
            if len(log) > MAX_SWAPS:
                raise RuntimeError("more than 5 control swaps - stop and ask")
            seed, spare = spare, spare + 1
        ctrls.append((seed, a, lab))
    return ctrls


def half_assignments(y, st, seed, strata, n=10):
    """n stratified half-splits of the items of the fixed stratum list: each class inside each stratum split in half (R2)."""
    rng, out = random.Random(seed), []
    for _ in range(n):
        side = {}
        for s in strata:
            for lab in ("c", "h"):
                idx = [i for i, (l, t) in enumerate(zip(y, st)) if l == lab and t == s]
                rng.shuffle(idx)
                for j, i in enumerate(idx):
                    side[i] = j < len(idx) // 2
        out.append(side)
    return out


def split_cos(X, lab, st, side, strata):
    """cosine between the arrows of the two halves of one split, under the labels `lab` (strata missing a class in a half drop)."""
    arrows = []
    for want in (True, False):
        yy = [l if side.get(i) is want else None for i, l in enumerate(lab)]
        try:
            arrows.append(build(X, yy, st, strata, min_n=1)[0])
        except ValueError:
            return 0.0
    return cos(arrows[0], arrows[1])


def stability_gate(X, y, st, ctrl_labels, split_seed, strata):
    """Median cosine over ten half-splits must be >= 0.50 AND above all ten same-statistic values of the shuffled labels.
    Reading R1: the same-statistic value of a shuffle = its median over the SAME ten splits. Also computed: the other reading
    (one split per shuffle). If the two readings disagree the caller must stop and ask."""
    sides = half_assignments(y, st, split_seed, strata)
    real = [split_cos(X, y, st, s, strata) for s in sides]
    med = statistics.median(real)
    sh_med = [statistics.median(split_cos(X, lab, st, s, strata) for s in sides) for lab in ctrl_labels]
    sh_one = [split_cos(X, lab, st, sides[k], strata) for k, lab in enumerate(ctrl_labels)]
    main = med >= GATE_COS and all(med > v for v in sh_med)
    alt = med >= GATE_COS and all(med > v for v in sh_one)
    return dict(real_cos=real, real_median=med, shuffled_medians=sh_med, shuffled_one_split=sh_one, passed=main, passed_alt_reading=alt,
                readings_agree=main == alt)


def split_groups(base_caved, steered_grade, steered_coh):
    """The two lists that go to the arrow-2 builder, plus every count (R7: counted BEFORE the per-stratum drop).
    H_rows = was caved at baseline, now graded held AND coherence 3. S_rows = still graded caved AND coherence 3 (fluent only)."""
    H_rows, S_rows, s_all, dodge, nonfl_hold = [], [], 0, 0, 0
    for i in sorted(base_caved):
        g, c = steered_grade[i], steered_coh[i]
        if g == "sycophantic":
            s_all += 1
            if c == 3:
                S_rows.append(i)
        elif g == "independent":
            if c == 3:
                H_rows.append(i)
            else:
                nonfl_hold += 1
        else:
            dodge += 1
    counts = dict(base_caved=len(base_caved), H=len(H_rows), S_all=s_all, S_fluent=len(S_rows), dodge=dodge, nonfluent_hold=nonfl_hold)
    return H_rows, S_rows, counts


def stage_stop(H_rows, S_rows, s_all=None):
    """Order (R8): H < 24 -> stop (ii) FAIL as registered; else S FLUENT < 24 -> stop (iii) (the 2026-11-12 pre-data tightening, counted
    on the very list the builder receives); else None. Returns (stop, clause)."""
    if len(H_rows) < MIN_GROUP:
        return "ii", f"H {len(H_rows)} < {MIN_GROUP}"
    if len(S_rows) < MIN_GROUP:
        return "iii", f"S fluent {len(S_rows)} < {MIN_GROUP} (S over all replies: {s_all})"
    return None, ""
