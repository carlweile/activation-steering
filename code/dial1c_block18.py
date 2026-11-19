"""Dial 1c side check: the giving-in direction at block 18 (prereg section 2; descriptive only, no controls).

    python dial1c_block18.py     # GPU ~2 min; writes results/dial1c_block18.json + results/dial1c_dir18.pt

d_GI18 = unit mean of (h_twin - h_framed) over the 640 answer + are-you-sure pairs, read at block 18 (hidden_states[19]),
last prompt token. Same 10 half-split stability gate as 1b (cos >= 0.80, seed 20261108, split by fact group).
alpha18 = 12.53 * NORM_18 / NORM_17 (NORM = mean residual norm of the 640 framed prompts at the read position).
Sanity: the block-17 direction recomputed in the SAME pass must match the stored d_GI (cos ~ 1).
"""
import json
import random

import torch

import dial1_lib as L
from dial1c_gen import index_check

RES = L.HERE / "results"
SEED, GATE, F_MAIN = 20261108, 0.80, 0.20


def unit(x):
    return x / x.norm()


def main():
    pairs = [p for p in L.rows("rebuild_pairs.jsonl") if p["kind"] in ("answer", "areyousure")]
    assert len(pairs) == 640
    tok, model = L.load()
    index_check(tok, model)
    cap = {}
    hk = model.model.layers[18].register_forward_hook(lambda m, i, o: cap.update(x=(o[0] if isinstance(o, tuple) else o)))
    with torch.no_grad():
        enc = tok([L.chat(tok, "Hello there")], return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[19]
    d18 = (cap["x"] - hs).abs().max().item()
    hk.remove()
    print("index check block 18, max abs diff", d18)
    if d18 != 0:
        raise SystemExit("block-18 index check failed - stop")
    cache = {}

    @torch.no_grad()
    def read(prompt):
        if prompt not in cache:
            e = tok([L.chat(tok, prompt)], return_tensors="pt").to("cuda")
            hs = model(**e, output_hidden_states=True).hidden_states
            cache[prompt] = (hs[18][0, -1].float().cpu(), hs[19][0, -1].float().cpu())   # block 17, block 18
        return cache[prompt]

    fr = [read(p["prompt"]) for p in pairs]
    tw = [read(p["twin"]) for p in pairs]
    out = {}
    for name, j in (("b17", 0), ("b18", 1)):
        diffs = torch.stack([t[j] - f[j] for t, f in zip(tw, fr)])
        norm = torch.stack([f[j] for f in fr]).norm(dim=-1).mean().item()
        grp = [p["group"] for p in pairs]
        gset = sorted(set(grp))
        rng = random.Random(SEED)
        halves = []
        for _ in range(10):
            hv = set(rng.sample(gset, len(gset) // 2))
            halves.append(unit(diffs[[i for i, g in enumerate(grp) if g in hv]].mean(0)))
        pw = [torch.dot(a, b).item() for i, a in enumerate(halves) for b in halves[i + 1:]]
        out[name] = dict(dir=unit(diffs.mean(0)), norm=norm, pairwise_min=min(pw), pairwise_median=sorted(pw)[len(pw) // 2])
    st = torch.load(RES / "dial1b_dirs.pt")
    cos17 = torch.dot(out["b17"]["dir"], st["gi_dir"].float()).item()
    alpha17 = F_MAIN * st["norm"]
    alpha18 = alpha17 * out["b18"]["norm"] / out["b17"]["norm"]
    gate = out["b18"]["pairwise_min"] >= GATE
    doc = dict(index_check_b18=d18, cos_recomputed_b17_vs_stored=cos17, norm17=out["b17"]["norm"], norm18=out["b18"]["norm"],
               alpha17=alpha17, alpha18=alpha18, pairwise_min_b17=out["b17"]["pairwise_min"], pairwise_min_b18=out["b18"]["pairwise_min"],
               pairwise_median_b18=out["b18"]["pairwise_median"], cos_b18_to_b17=torch.dot(out["b18"]["dir"], st["gi_dir"].float()).item(),
               gate_line=GATE, gate="PASS" if gate else "FAIL")
    (RES / "dial1c_block18.json").write_text(json.dumps(doc, indent=1))
    torch.save(dict(dir18=out["b18"]["dir"], alpha18=alpha18), RES / "dial1c_dir18.pt")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
