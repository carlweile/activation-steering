"""Dial 1b: the giving-in direction, its stability gate and 10 shuffled-label controls (prereg section 1 + 3).

    python dial1b_build.py        # CPU only; reads results/dial1_rebuild_pairs.pt (card C per-pair differences)

Writes results/dial1b_build.json (numbers) and results/dial1b_dirs.pt (d_GI, c0..c9, norm unit).
"""
import datetime
import json
import random

import torch

from dial1_lib import HERE, rows

RES = HERE / "results"
GATE = 0.80
SEED = 20261108
KINDS = ("answer", "areyousure")


def unit(x):
    return x / x.norm()


def main():
    st = torch.load(RES / "dial1_rebuild_pairs.pt")
    pairs = {p["id"]: p for p in rows("rebuild_pairs.jsonl")}
    ids, groups, diffs = st["ids"], st["groups"], st["diffs"].float()
    keep = [i for i, pid in enumerate(ids) if pairs[pid]["kind"] in KINDS]
    kinds = [pairs[ids[i]]["kind"] for i in keep]
    gi = diffs[keep]
    assert len(keep) == 640 and kinds.count("answer") == 360 and kinds.count("areyousure") == 280
    mean_diff = gi.mean(0)
    d = unit(mean_diff)

    def sub(idx):
        return unit(gi[idx].mean(0))

    rng = random.Random(SEED)
    grp = [groups[i] for i in keep]
    gset = sorted(set(grp))
    halves = []
    for _ in range(10):
        hv = set(rng.sample(gset, len(gset) // 2))
        halves.append(sub([j for j, g in enumerate(grp) if g in hv]))
    pw = [torch.dot(a, b).item() for i, a in enumerate(halves) for b in halves[i + 1:]]
    gate = min(pw) >= GATE
    # controls: sign flipped for exactly half the pairs, balanced within kind
    ctrl, ctrl_raw = [], []
    for k in range(10):
        r = random.Random(k)
        sign = torch.ones(len(keep))
        for kind in KINDS:
            idx = [j for j, x in enumerate(kinds) if x == kind]
            for j in r.sample(idx, len(idx) // 2):
                sign[j] = -1.0
        m = (gi * sign.unsqueeze(1)).mean(0)
        ctrl_raw.append(m.norm().item())
        ctrl.append(unit(m))
    old_c = torch.load(RES / "dial1_rebuild_dir.pt")
    old_0b = torch.load(RES / "dial1_dom_dir.pt")
    ans = sub([j for j, x in enumerate(kinds) if x == "answer"])
    sure = sub([j for j, x in enumerate(kinds) if x == "areyousure"])
    doc = dict(date=str(datetime.date.today()), n_pairs=len(keep), kinds={k: kinds.count(k) for k in KINDS},
               raw_norm_mean_diff=mean_diff.norm().item(), strength_unit_norm=old_c["norm"],
               half_sample_pairwise_cosines=pw, pairwise=dict(min=min(pw), median=sorted(pw)[len(pw) // 2], max=max(pw)),
               gate_line=GATE, gate="PASS" if gate else "FAIL",
               cos_to_cardC_direction=torch.dot(d, old_c["rebuild_dir"].float()).item(),
               cos_to_run0b_direction=torch.dot(d, old_0b["dom_dir"].float()).item(),
               cos_answer_vs_areyousure=torch.dot(ans, sure).item(),
               control_raw_norms=ctrl_raw, control_cos_to_real=[torch.dot(c, d).item() for c in ctrl])
    (RES / "dial1b_build.json").write_text(json.dumps(doc, indent=1))
    torch.save(dict(gi_dir=d, controls=torch.stack(ctrl), norm=old_c["norm"]), RES / "dial1b_dirs.pt")
    print(json.dumps({k: v for k, v in doc.items() if k != "half_sample_pairwise_cosines"}, indent=1))
    if not gate:
        raise SystemExit("stability gate FAILED - no sweep")


if __name__ == "__main__":
    main()
