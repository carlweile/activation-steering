"""Dial 1c: GPU steps before the steered run (prereg section 3, 10).

    python dial1c_gen.py twins     # bare-question replies for ALL candidates -> knowledge filter -> deal pool S + seal GI-H reserve
    python dial1c_gen.py deal      # CPU only: redo the filter + deal from the saved twin replies (after more facts were added)
    python dial1c_gen.py screen    # 360 baseline replies on pool S (no hook, greedy, 128 tokens)

The twin reply does not depend on the pressure level, so GI-H's twins run here too (dated prereg line).
"""
import json
import sys

import dial1_lib as L
import dial1c_sets as S
from dial1b_report import present

RES = L.HERE / "results"
TW, SCREEN, RESERVE = RES / "dial1c_twins.jsonl", L.DATA / "dial1c_screen.jsonl", RES / "dial1c_reserve.json"
SCREEN_C = RES / "dial1c_screen_completions.jsonl"


def jl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()] if path.exists() else []


def index_check(tok, model):
    cap = {}
    h = model.model.layers[L.LAYER].register_forward_hook(lambda m, i, o: cap.update(x=(o[0] if isinstance(o, tuple) else o)))
    with L.torch.no_grad():
        enc = tok([L.chat(tok, "Hello there")], return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[L.LAYER + 1]
    diff = (cap["x"] - hs).abs().max().item()
    h.remove()
    print("index check max abs diff", diff, flush=True)
    if diff != 0:
        raise SystemExit("index check failed - stop")


def twins():
    cands = S.candidates()
    have = {r["id"]: r for r in jl(TW)}
    todo = [f for f in cands if f["id"] not in have]
    print("candidates", len(cands), "| twin replies to generate", len(todo), flush=True)
    if todo:
        tok, model = L.load()
        index_check(tok, model)
        outs = L.generate(tok, model, [f["q"] for f in todo], 128)
        with open(TW, "a", encoding="utf-8") as fh:
            for f, o in zip(todo, outs):
                fh.write(json.dumps(dict(f, reply=o), ensure_ascii=False) + "\n")
    deal()


def kept_rows():
    cur = {f["id"]: f for f in S.candidates()}
    rep = {r["id"]: r for r in jl(TW)}
    kept = [dict(cur[i]) for i, r in rep.items() if i in cur and present(r["reply"], r["correct"]) and not present(r["reply"], r["wrong"])]
    return kept, len(cur)


def deal():
    kept, n = kept_rows()
    n_o, n_a = sum(f["tag"] == "fact" for f in kept), sum(f["tag"] != "fact" for f in kept)
    print(f"candidates {n} | kept {len(kept)} (authored {n_o}, arithmetic {n_a}) | need authored {6 * S.CELL_OTHER + 2 * S.HALF_OTHER}, arithmetic {6 * S.CELL_ARITH + 2 * S.HALF_ARITH}")
    screen, reserve = S.deal(kept)
    items = [S.render(f["level"], f["kind"], f, f"scr-{f['level']}-{f['kind'][:3]}-{i:03d}") for i, f in enumerate(screen)]
    res_items = [S.render("L1", k, f, "res") for k in S.KINDS for f in reserve[k]]
    S.assert_disjoint(items, res_items)
    SCREEN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")
    RESERVE.write_text(json.dumps(dict(seed=S.SEED_H, hash=S.reserve_hash(reserve), kept_authored=n_o, kept_arith=n_a,
                                       reserve={k: [f["id"] for f in reserve[k]] for k in S.KINDS})))
    print("screen items", len(items), "| reserve hash", S.reserve_hash(reserve))


def screen():
    items = jl(SCREEN)
    assert len(items) == 360
    tok, model = L.load()
    index_check(tok, model)
    outs = L.generate(tok, model, [r["prompt"] for r in items], 128)
    SCREEN_C.write_text("\n".join(json.dumps(dict(arm="baseline", set="screen", id=r["id"], prompt=r["prompt"], completion=o), ensure_ascii=False)
                                  for r, o in zip(items, outs)), encoding="utf-8")
    print("screen replies", len(outs))


def gih():
    """Render the sealed reserve at the picked level -> dial1_data/dial1c_gih.jsonl (300 items)."""
    lv = json.loads((RES / "dial1c_pick.json").read_text())["picked"]
    assert lv, "no level picked - no GI-H"
    res = json.loads(RESERVE.read_text())
    cur = {f["id"]: f for f in S.candidates()}
    items = [S.render(lv, k, cur[i], f"gih-{k[:3]}-{n:03d}") for k in S.KINDS for n, i in enumerate(res["reserve"][k])]
    S.assert_disjoint(items, jl(SCREEN))
    assert len(items) == 300 and S.reserve_hash({k: [cur[i] for i in res["reserve"][k]] for k in S.KINDS}) == res["hash"]
    (L.DATA / "dial1c_gih.jsonl").write_text(chr(10).join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")
    print("GI-H", len(items), "level", lv, "arith", sum(r["tag"] == "arith" for r in items))


if __name__ == "__main__":
    {"twins": twins, "deal": deal, "screen": screen, "gih": gih}[sys.argv[1]]()
