"""Dial 1d GPU helpers: the two-hook class (blocks 17 and 18), the index check on BOTH blocks, the state read, the strengths.

dial1_lib.Dial hooks ONE block (module-global LAYER); this class hooks two, each with its own mode ("add", unit arrow, alpha) or None.
Strength unit (prereg 3.5): alpha17 = f * 62.637 (stored norm); alpha18 = f * 62.637 * 62.8176 / 57.4622 (the 1c block-18 ratio rule).
"""
import json

import torch

import dial1_lib as L

RES = L.HERE / "results"
BLOCKS = (17, 18)


def norms():
    st = torch.load(RES / "dial1b_dirs.pt")
    b18 = json.loads((RES / "dial1c_block18.json").read_text())
    return float(st["norm"]), b18["norm18"] / b18["norm17"]


def alpha17(f):
    return f * norms()[0]


def alpha18(f):
    n, ratio = norms()
    return f * n * ratio


class TwoDial:
    def __init__(self, model):
        self.mode = {17: None, 18: None}
        self.h = [model.model.layers[b].register_forward_hook(self._fn(b)) for b in BLOCKS]

    def _fn(self, b):
        def fn(_m, _inp, out):
            m = self.mode[b]
            if m is None:
                return None
            x = out[0] if isinstance(out, tuple) else out
            x = x + m[1] * m[0].to(x.device, x.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return fn

    def set(self, m17=None, m18=None):
        """m17 / m18 = (unit arrow, alpha) or None."""
        self.mode = {17: m17, 18: m18}


def index_check(tok, model):
    """Max abs diff between each block's hooked output and hidden_states[block + 1], no push on. Must be 0 on both blocks."""
    out = {}
    for b in BLOCKS:
        cap = {}
        h = model.model.layers[b].register_forward_hook(lambda m, i, o: cap.update(x=(o[0] if isinstance(o, tuple) else o)))
        with torch.no_grad():
            enc = tok([L.chat(tok, "Hello there")], return_tensors="pt").to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[b + 1]
        out[b] = (cap["x"] - hs).abs().max().item()
        h.remove()
    print("index check max abs diff", out, flush=True)
    if any(v != 0 for v in out.values()):
        raise SystemExit(f"index check failed {out} - stop")
    return out


@torch.no_grad()
def read_states(tok, model, prompts, block, bs=16):
    """Output of `block` at the LAST prompt token (left padding), whatever modes the hooks are in. float32 on CPU."""
    cap, vecs = {}, []
    h = model.model.layers[block].register_forward_hook(lambda m, i, o: cap.update(x=(o[0] if isinstance(o, tuple) else o)))
    try:
        for b in L.batches(prompts, bs):
            enc = tok([L.chat(tok, p) for p in b], return_tensors="pt", padding=True).to("cuda")
            model(**enc)
            vecs.append(cap["x"][:, -1].float().cpu())
    finally:
        h.remove()
    return torch.cat(vecs)
