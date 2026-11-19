"""Shared pieces (CPU): pairs, cross-fit folds, per-template directions, stability gate, shuffled-label controls.
Prereg: results/prereg_dial1_percell_2026-11-04.md PART 2. +f = toward the neutral twin; alpha = f x NORM."""
import random

import torch

import dial1_lib as L

NORM = 62.63673782348633  # strength unit of run 0b (dial1_rebuild_dir.pt "norm")
SEED = 20261104
ARITH = {"mul", "add", "days", "hours"}
GROUPS = {"FB": ("feedback",), "AS": ("answer", "areyousure"), "joint": ("feedback", "answer", "areyousure")}
KINDS = ("feedback", "answer", "areyousure")
GATE = 0.80


def pairs():
    return L.rows("rebuild_pairs.jsonl")


def make_folds(ps):
    """group -> 'A'|'B'; stratified by kind (and arithmetic vs other for answer/areyousure), seeded."""
    rng, strata, fold = random.Random(SEED), {}, {}
    for p in ps:
        if p["group"] not in fold and p["group"] not in {g for s in strata.values() for g in s}:
            key = (p["kind"], p["kind"] != "feedback" and p.get("tag") in ARITH)
            strata.setdefault(key, []).append(p["group"])
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold[g] = "AB"[i % 2]
    return fold


def other(f):
    return "B" if f == "A" else "A"


def unit(x):
    return x / x.norm()


def idx_of(ps, fold, kinds, train_fold):
    return [i for i, p in enumerate(ps) if p["kind"] in kinds and fold[p["group"]] == train_fold]


def direction(diffs, idx):
    return unit(diffs[idx].mean(0))


def stability(diffs, ps, idx, n=10, seed=SEED):
    """10 random half-samples by group -> all 45 pairwise cosines."""
    groups = sorted({ps[i]["group"] for i in idx})
    rng, halves = random.Random(seed), []
    for _ in range(n):
        hv = set(rng.sample(groups, len(groups) // 2))
        halves.append(direction(diffs, [i for i in idx if ps[i]["group"] in hv]))
    cs = [torch.dot(a, b).item() for i, a in enumerate(halves) for b in halves[i + 1:]]
    return dict(min=min(cs), median=sorted(cs)[len(cs) // 2], max=max(cs), gate="PASS" if min(cs) >= GATE else "FAIL")


def control_dir(diffs, idx, seed):
    """shuffled-label control: random sign flip of each pair's difference before averaging."""
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (len(idx),), generator=g) * 2 - 1).float().unsqueeze(1)
    return unit((diffs[idx] * s).mean(0))


def build_all(ps, diffs, fold):
    """{train_fold: {group_name: unit direction}} + stability stats."""
    dirs, stab = {}, {}
    for tf in "AB":
        for name, kinds in GROUPS.items():
            ix = idx_of(ps, fold, kinds, tf)
            dirs.setdefault(tf, {})[name] = direction(diffs, ix)
            stab[f"{name}_train{tf}"] = dict(n_pairs=len(ix), **stability(diffs, ps, ix))
    return dirs, stab
