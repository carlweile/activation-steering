"""Dial 1d: the new facts, the GI-T deal (n = 300, or the n = 174 fallback) and the role-disjointness assertion.

Prereg results/prereg_dial1d_2026-11-11.md 3.1. Pure functions, CPU only (the GPU side is dial1d_run.py).
Roles: BUILD = dial1c_screen.jsonl (360), FEEDBACK = dial1c_gih.jsonl (300), TEST = GI-T (300 or 174), all disjoint by prompt AND fact.
"""
import hashlib
import json
import random

from dial1_lib import DATA, HERE, rows
from dial1_rebuild_sets import forbidden, norm
from dial1b_sets import clash, used_by_build
from dial1b_report import present
import dial1c_sets as S

RES = HERE / "results"
SEED_T, SEED_SPLIT, SEED_POOL = 20261140, 20261141, 20261142     # deal / half-splits / pool shuffle (readings file, R5)
LEVEL = "L2"
NEED_NEW = 87                                                    # new facts that must pass for n = 300 (prereg 3.1)
SIZES = {300: dict(auth=105, arith=45), 174: dict(auth=61, arith=26)}   # per half


def new_rows():
    """All rows the writers wrote, both rounds (dropped ones too: none may re-enter). Data file dial1d_facts.py."""
    from dial1d_facts import FACTS, FACTS2
    return [dict(tag="fact", key=norm(q), q=q, correct=c, wrong=w, id="d" + hashlib.sha1(norm(q).encode()).hexdigest()[:8])
            for q, c, w in FACTS + FACTS2]


def review_dropped():
    """Ids of rows either fresh reviewer flagged (results/dial1d_facts_review.json)."""
    rev = json.loads((RES / "dial1d_facts_review.json").read_text(encoding="utf-8"))
    return set(rev["dropped_ids"])


def cpu_excluded(nrows, stored_q=None, stored_i=None):
    """Ids of rows whose question (normalised) or id is already stored anywhere (cands, build pairs, 1b sets, held-out texts), or that fail
    the 1c `clash` rule (question or entity key inside a held-out text, substring included)."""
    stored_q = stored_questions() if stored_q is None else stored_q
    stored_i = stored_ids() if stored_i is None else stored_i
    held, build = [norm(t) for t in forbidden()], used_by_build()
    return {r["id"] for r in nrows if norm(r["q"]) in stored_q or r["id"] in stored_i or clash(r, held, build)}


def usable_new():
    """Rows that go to the knowledge filter: not flagged by a reviewer, not already stored."""
    nr = new_rows()
    bad = review_dropped() | cpu_excluded(nr)
    return [r for r in nr if r["id"] not in bad]


def stored_questions():
    """Normalised questions of every stored candidate (authored + arithmetic) plus 1b sets, build pairs, held-out texts."""
    tw = {norm(json.loads(l)["q"]) for l in open(RES / "dial1c_twins.jsonl", encoding="utf-8") if l.strip()}
    return tw | {norm(r["twin"]) for f in ("dial1c_screen.jsonl", "dial1c_gih.jsonl") for r in rows(f)} \
        | S.earlier_questions() | used_by_build() | {norm(t) for t in forbidden()}


def stored_ids():
    return {json.loads(l)["id"] for l in open(RES / "dial1c_twins.jsonl", encoding="utf-8") if l.strip()}


def check_new_rows(nrows, stored_q=None, stored_i=None):
    """(f) of the plan: no new question equals a stored one (normalised), no new id equals a stored id, no repeat inside the new rows."""
    stored_q = stored_questions() if stored_q is None else stored_q
    stored_i = stored_ids() if stored_i is None else stored_i
    seen = set()
    for r in nrows:
        q = norm(r["q"])
        assert q not in stored_q, ("new question already stored", r["q"])
        assert r["id"] not in stored_i, ("new id equals stored id", r["id"])
        assert q not in seen, ("duplicate inside new rows", r["q"])
        assert norm(r["correct"]) != norm(r["wrong"]), ("correct == wrong", r["q"])
        seen.add(q)


def stored_unused():
    """Unused kept stored candidates (123 authored / 190 arithmetic), recomputed by the Appendix A rule."""
    tw = [json.loads(l) for l in open(RES / "dial1c_twins.jsonl", encoding="utf-8") if l.strip()]
    kept = [t for t in tw if present(t["reply"], t["correct"]) and not present(t["reply"], t["wrong"])]
    used = {g["src"] for f in ("dial1c_gih.jsonl", "dial1c_screen.jsonl") for g in rows(f)}
    cur = {f["id"]: f for f in S.candidates()}
    return [dict(cur[t["id"]]) for t in kept if t["id"] not in used and t["id"] in cur], len(kept)


def kept_new(twins_path=None):
    """New rows that passed the plain-question knowledge filter (1c rule): correct present AND near-miss absent."""
    p = twins_path or RES / "dial1d_twins.jsonl"
    cur = {f["id"]: f for f in usable_new()}
    out = []
    for l in open(p, encoding="utf-8"):
        if l.strip():
            t = json.loads(l)
            if t["id"] in cur and present(t["reply"], t["correct"]) and not present(t["reply"], t["wrong"]):
                out.append(dict(cur[t["id"]]))
    return out


def deal_test(unused, new_kept, n_new_pass_needed=NEED_NEW, seed=SEED_T):
    """Return (n, {kind: [facts]}). n = 300 if len(new_kept) >= NEED_NEW, else the n = 174 fallback from stored items only."""
    n = 300 if len(new_kept) >= n_new_pass_needed else 174
    sz = SIZES[n]
    auth = [f for f in unused if f["tag"] == "fact"]
    ari = [f for f in unused if f["tag"] != "fact"]
    pool = sorted(auth + (new_kept if n == 300 else []), key=lambda f: f["id"])
    ari = sorted(ari, key=lambda f: f["id"])
    if len(pool) < 2 * sz["auth"] or len(ari) < 2 * sz["arith"]:
        raise SystemExit(f"SHORTFALL: authored {len(pool)}/{2 * sz['auth']}, arithmetic {len(ari)}/{2 * sz['arith']}")
    r = random.Random(seed)
    r.shuffle(pool)
    r.shuffle(ari)
    deal = {}
    for kind in S.KINDS:
        half = [pool.pop() for _ in range(sz["auth"])] + [ari.pop() for _ in range(sz["arith"])]
        r.shuffle(half)
        deal[kind] = half
    return n, deal


def render_test(deal):
    return [S.render(LEVEL, k, f, f"git-{k[:3]}-{i:03d}") for k in S.KINDS for i, f in enumerate(deal[k])]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def assert_roles_disjoint(build, feedback, test, n_expected, unused=None, new_kept=None):
    """Counts 360 / 300 / n; pairwise disjoint by prompt AND by fact (src and normalised question); GI-T facts come from the unused
    stored set or the kept new set; the 1b / build / held-out checks of dial1c_sets.assert_disjoint still hold."""
    assert (len(build), len(feedback), len(test)) == (360, 300, n_expected), (len(build), len(feedback), len(test))
    groups = dict(build=build, feedback=feedback, test=test)
    for a in groups:
        for b in groups:
            if a < b:
                assert not {r["prompt"] for r in groups[a]} & {r["prompt"] for r in groups[b]}, ("prompt overlap", a, b)
                assert not {r["src"] for r in groups[a]} & {r["src"] for r in groups[b]}, ("fact (src) overlap", a, b)
                assert not {norm(r["twin"]) for r in groups[a]} & {norm(r["twin"]) for r in groups[b]}, ("fact (question) overlap", a, b)
    assert len({r["src"] for r in test}) == len(test) and len({norm(r["twin"]) for r in test}) == len(test), "fact repeats inside TEST"
    if unused is not None:
        ok_ids = {f["id"] for f in unused} | {f["id"] for f in (new_kept or [])}
        bad = [r["src"] for r in test if r["src"] not in ok_ids]
        assert not bad, ("TEST fact not in the unused-kept or new-kept set", bad[:3])
    S.assert_disjoint(build, feedback, test)


def load_roles():
    return rows("dial1c_screen.jsonl"), rows("dial1c_gih.jsonl")
