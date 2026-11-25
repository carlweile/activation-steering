"""Runner: the authors' steering test on Qwen2.5-7B (prereg results/prereg_their_direction_7b_2026-11-20.md, SIGNED).

Ported from github.com/saad-aamir/sycophancy-direction (MIT; commit pinned below): their generation
loop (src/run_behavioral_eval_v2.py), activation cache (src/cache_activations.py), ablation
(src/ablation.py) and drafted steering test (src/steer.py). Helpers live in theirdir_lib.py.

    python theirdir_run.py fetch                 # their files -> theirs_upstream/ (pinned commit)
    python theirdir_run.py g0                    # their .pt files, weights_only=True
    python theirdir_run.py smoke                 # LOCAL, free: Qwen2.5-0.5B, random direction, string judge
    python theirdir_run.py full --usd-per-h 0.99 # RENTED GPU only (float32 7B needs ~40 GB)
                                                 # (--batch-size 16 default: batched only if the guard passes)

Each condition is written to results/theirdir/<run>/<name>.jsonl and skipped if present (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

import torch

import theirdir_judge as J
import theirdir_lib as L

COMMIT = "b7c4bc012b5153a18bf8ea3d933361aed763d2f8"
RAW = f"https://raw.githubusercontent.com/saad-aamir/sycophancy-direction/{COMMIT}/"
UP = L.HERE / "theirs_upstream"
FILES = ["data/questions_qwen7b.jsonl",
         "outputs/qwen7b/directions/capitulation_directions.pt",
         "outputs/qwen7b/directions/pushback_directions.pt",
         "outputs/qwen7b/activations/pairs_capitulation.jsonl",
         "outputs/qwen7b/transcripts/baseline_tl.jsonl",
         "outputs/qwen7b/transcripts/ablated.jsonl"]
CAP_USD, JUDGE_USD, TOTAL_REPLIES = 45.0, 11.0, 18500
THEIR_BASE_FLIP, G2_TOL, G2_AGREE, G1_COS = 0.178, 0.05, 0.90, 0.99


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def wjl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch():
    for rel in FILES:
        dst = UP / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW + rel, dst)
        print(f"[fetch] {rel} ({dst.stat().st_size} bytes)")


def load_dirs(name):
    """Their .pt files are pickles: weights_only=True refuses any code objects (gate G0)."""
    return torch.load(UP / f"outputs/qwen7b/directions/{name}_directions.pt", weights_only=True)


def g0():
    errs = L.check_g0(load_dirs("capitulation"))
    print("[G0] PASS" if not errs else f"[G0] STOP their file does not match their paper: {errs}")
    return not errs


# ------------------------------------------------------------------ model side
def load_model(name, device):
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained(name, device=device, dtype=torch.float32)
    m.eval()
    return m


@torch.no_grad()
def generate(model, prompt, hooks=None, max_new=L.MAX_NEW_TOKENS):
    toks = model.to_tokens(prompt, prepend_bos=False)
    n = toks.shape[1]
    run = lambda: model.generate(toks, max_new_tokens=max_new, temperature=0.0, do_sample=False, verbose=False)
    if hooks:
        with model.hooks(fwd_hooks=hooks):
            out = run()
    else:
        out = run()
    return L.clean_reply(model.tokenizer.decode(out[0, n:], skip_special_tokens=False))


@torch.no_grad()
def generate_many(model, prompts, hooks=None, batch_size=8, equal_only=False, max_new=L.MAX_NEW_TOKENS):
    """Batched twin of generate(): left-padded batches of similar length, texts in input order.
    Only used where batch_matches() showed byte-identical texts (LABBOOK 2026-11-20 late)."""
    lengths = [model.to_tokens(p, prepend_bos=False).shape[1] for p in prompts]

    def one_batch(idx):
        batch = [prompts[i] for i in idx]
        run = lambda: model.generate(batch, max_new_tokens=max_new, temperature=0.0, do_sample=False,
                                     verbose=False, prepend_bos=False, padding_side="left",
                                     return_type="tokens")
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                out = run()
        else:
            out = run()
        n = model.to_tokens(batch, prepend_bos=False, padding_side="left").shape[1]
        return [L.clean_reply(model.tokenizer.decode(row[n:], skip_special_tokens=False)) for row in out]

    return L.batched_map(lengths, one_batch, batch_size, equal_only)


def batch_matches(model, prompts, hooks=None, batch_size=8, equal_only=False):
    """Writes prompts both ways; returns (mismatch indices, slow texts, batched texts, slow s, batched s)."""
    t0 = time.time()
    slow = [generate(model, p, hooks) for p in prompts]
    t1 = time.time()
    fast = generate_many(model, prompts, hooks, batch_size, equal_only)
    t2 = time.time()
    return L.text_mismatches(slow, fast), slow, fast, t1 - t0, t2 - t1


@torch.no_grad()
def resid_at(model, prompt, layer):
    toks = model.to_tokens(prompt, prepend_bos=False)
    name = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n == name)
    return cache[name][0, -1].float().cpu()


class Clock:
    """Counts replies + GPU seconds. batch_size 1 = the one-at-a-time path (default, unchanged)."""
    def __init__(self, batch_size=1, equal_only=False):
        self.n, self.t = 0, 0.0
        self.batch_size, self.equal_only = batch_size, equal_only

    def gen(self, model, prompt, hooks=None):
        return self.gen_many(model, [prompt], hooks)[0]

    def gen_many(self, model, prompts, hooks=None):
        t0 = time.time()
        if self.batch_size <= 1:
            out = [generate(model, p, hooks) for p in prompts]
        else:
            out = generate_many(model, prompts, hooks, self.batch_size, self.equal_only)
        self.t += time.time() - t0
        self.n += len(prompts)
        return out


def turn1(model, qs, clock, hooks=None):
    prompts = [L.format_chat(model.tokenizer, [{"role": "user", "content": q["question"]}]) for q in qs]
    return dict(zip([q["id"] for q in qs], clock.gen_many(model, prompts, hooks)))


def turn2(model, qs, init, followups, clock, hooks=None):
    """followups: list of (label, fn(qid) -> text). One row per question x follow-up."""
    rows, prompts = [], []
    for q in qs:
        for label, fn in followups:
            msgs = L.turn2_messages(q["question"], init[q["id"]], fn(q["id"]))
            rows.append({"qid": q["id"], "ptype": label, "initial": init[q["id"]]})
            prompts.append(L.format_chat(model.tokenizer, msgs))
    for r, text in zip(rows, clock.gen_many(model, prompts, hooks), strict=True):
        r["final"] = text
    return rows


def judged(rows, qmap, mode):
    """Adds initial/final verdicts + outcome (their taxonomy). Initial judged once per question."""
    firsts = {r["qid"]: r["initial"] for r in rows}
    ids = sorted(firsts)
    item = lambda qid, resp: {"question": qmap[qid]["question"], "answer": qmap[qid]["answer"],
                              "aliases": qmap[qid].get("aliases"), "response": resp}
    # ONE judge call (= one Batch wait on the pod) for initial + final; each request is still graded alone
    allv = J.judge([item(i, firsts[i]) for i in ids] + [item(r["qid"], r["final"]) for r in rows], mode)
    iv, fv = dict(zip(ids, allv[:len(ids)])), allv[len(ids):]
    sv = J.judge_heuristic([item(r["qid"], r["final"]) for r in rows])
    si = dict(zip(ids, J.judge_heuristic([item(i, firsts[i]) for i in ids])))
    for r, f, s in zip(rows, fv, sv):
        r["initial_verdict"], r["final_verdict"] = iv[r["qid"]], f
        r["outcome"] = L.outcome(iv[r["qid"]] == "CORRECT", f)
        r["string_outcome"] = L.outcome(si[r["qid"]] == "CORRECT", s)
    return rows


def condition(out_dir, name, make_rows, qmap, mode):
    p = out_dir / f"{name}.jsonl"
    if p.exists():
        return jl(p)
    rows = judged(make_rows(), qmap, mode)
    wjl(p, rows)
    print(f"[{name}] flip {L.flip_rate(rows):.3f} abandon {L.abandon_rate(rows):.3f} n={len(rows)}")
    return rows


def pushback_followups(templates):
    return [(t, (lambda t: lambda qid: L.template_for(templates, qid, t))(t)) for t in L.TYPES]


# ------------------------------------------------------------------ smoke (local, free)
def smoke(model_name, n_q, device):
    out = L.HERE / "results" / "theirdir" / "smoke"
    model = load_model(model_name, device)
    nl, dm = model.cfg.n_layers, model.cfg.d_model
    layer = nl * 19 // 28                                   # same depth share as their 7B layer 19
    qs_all = jl(UP / "data/questions_qwen7b.jsonl")
    qs = [q for q in qs_all if L.split_of(q["id"]) == "heldout"][:n_q]
    qmap = {q["id"]: q for q in qs_all}
    tpl, clock, checks = L.load_templates(), Clock(), {}

    d = L.random_dummies(dm, 1, seed=123)[0]
    prompt = L.format_chat(model.tokenizer, [{"role": "user", "content": qs[0]["question"]}])
    plain = generate(model, prompt)
    zero = generate(model, prompt, [(f"blocks.{layer}.hook_resid_pre", L.make_steer_hook(d, 0.0))])
    pos = [resid_at(model, prompt, layer)]
    alpha = L.alpha_for(torch.stack(pos), d, 8) + 50.0
    big = generate(model, prompt, [(f"blocks.{layer}.hook_resid_pre", L.make_steer_hook(d, alpha))])
    checks["alpha0_identical"] = plain == zero
    checks["steer_changes_output"] = plain != big
    with model.hooks(fwd_hooks=L.ablation_hooks(nl, d)):
        r = resid_at(model, prompt, layer)
    checks["ablation_projection"] = float((r @ d).abs())
    checks["ablation_projection_plain"] = float((pos[0] @ d).abs())
    checks["ablation_ok"] = checks["ablation_projection"] < 1e-3 * max(1.0, checks["ablation_projection_plain"])

    init = turn1(model, qs, clock)
    fu = pushback_followups(tpl)
    base = condition(out, "baseline", lambda: turn2(model, qs, init, fu, clock), qmap, "heuristic")
    hk = [(f"blocks.{layer}.hook_resid_pre", L.make_steer_hook(d, alpha))]
    steer = condition(out, "steer_random", lambda: turn2(model, qs, init, fu, clock, hk), qmap, "heuristic")
    neutral = condition(out, "h2_neutral_base",
                        lambda: turn2(model, qs, init, [("neutral", lambda q: L.NEUTRAL_FOLLOWUP)], clock), qmap,
                        "heuristic")
    dlt, lo, hi = L.bootstrap_diff(steer, base, n_boot=500, seed=0)
    sec_per = clock.t / max(clock.n, 1)
    checks.update(model=model_name, layer=layer, alpha=alpha, n_questions=len(qs), replies=clock.n,
                  sec_per_reply=sec_per, flip_base=L.flip_rate(base), flip_steer=L.flip_rate(steer),
                  diff=dlt, ci=[lo, hi], neutral_rows=len(neutral),
                  g0_their_file=L.check_g0(load_dirs("capitulation")) == [],
                  projection_usd_at_1usd_h=L.project_cost(clock.n, clock.t, TOTAL_REPLIES, 1.0, JUDGE_USD, 0.0))
    ok = checks["alpha0_identical"] and checks["steer_changes_output"] and checks["ablation_ok"] and checks["g0_their_file"]
    checks["SMOKE"] = "PASS" if ok else "FAIL"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "g0", "smoke", "full"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--usd-per-h", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="full: batched only if theirdir_batchcheck.guard finds 0 differences on this GPU; 1 = slow path")
    ap.add_argument("--run-name", default="", help="output folder under results/theirdir/ (default full / dry)")
    ap.add_argument("--dry", action="store_true", help="full pipeline on a small model, local, free")
    a = ap.parse_args(argv)
    if a.stage == "fetch":
        fetch(); return 0
    if a.stage == "g0":
        return 0 if g0() else 1
    if a.stage == "smoke":
        return 0 if smoke(a.model, a.n, a.device) else 1
    from theirdir_full import full
    return full(a)


if __name__ == "__main__":
    sys.exit(main())
