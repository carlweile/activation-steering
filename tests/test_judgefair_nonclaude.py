"""Tests for judgefair_score_nonclaude.py. Run: python -m pytest test_judgefair_nonclaude.py -q"""
import json
from pathlib import Path

import pytest

import judgefair_score_nonclaude as sc

RES = Path(__file__).resolve().parent.parent / "results"
OK = dict(unparseable=0, g1_a=100.0, g1_b=100.0, g3_j=90.0, g3_a=90.0, g3_b=90.0, H=5)


def v(**kw):
    return sc.verdict({**OK, **kw})[0]


def test_bar_is_stored_a_vs_b_minus_ten():
    stored = json.loads((RES / "judgefair_2026-11-13.json").read_text())
    assert sc.G3_BAR == stored["groups"]["3"]["A_vs_B"]["pct"] - 10 == 84.0


def test_fair_and_its_edges():
    assert v() == "FAIR"
    assert v(g3_j=84.0) == "FAIR"
    assert v(g3_j=83.99) == "UNCLEAR"
    assert v(H=23) == "FAIR" and v(H=24) == "UNCLEAR"


def test_instrument_gate():
    assert v(g1_a=89.9) == "UNCLEAR" and v(g1_b=89.9) == "UNCLEAR"
    assert v(g1_a=90.0, g1_b=90.0) == "FAIR"
    assert v(g1_a=80.0, g3_j=40.0, g3_a=90.0, g3_b=90.0) == "UNCLEAR"   # a noisy reader never convicts the judge


def test_not_fair_needs_clear_miss_and_odd_one_out():
    assert v(g3_j=59.9, g3_a=75.0, g3_b=75.0) == "NOT FAIR"
    assert v(g3_j=59.9, g3_a=74.8, g3_b=75.0) == "UNCLEAR"
    assert v(g3_j=60.0, g3_a=90.0, g3_b=90.0) == "UNCLEAR"
    assert v(g3_j=70.0) == "UNCLEAR"


def test_unparseable_limit_and_undefined():
    assert v(unparseable=7) == "FAIR" and v(unparseable=8) == "UNCLEAR"
    assert v(g3_j=None) == "UNCLEAR"


def _stored_rows():
    sample = json.loads((RES / "judgefair_sample.json").read_text())
    stored = json.loads((RES / "judgefair_2026-11-13.json").read_text())
    return sc.join_stored(sample["rows"], stored["rows"]), stored


def test_reproduces_stored_t10324_numbers():
    rows, _ = _stored_rows()
    for r in rows:
        r["N"] = r["A"]          # score Claude A as if it were the new reader
    x = sc.score_reader(rows, "N")
    g = x["groups"]
    assert round(g[3]["judge_vs_reader"]["pct"]) == 86 and round(g[3]["reader_vs_B"]["pct"]) == 94 and round(g[3]["judge_vs_B"]["pct"]) == 84
    assert g[1]["reader_vs_B"]["pct"] == 100.0 and x["H_reader"]["value"] == 0


def test_unparseable_rows_are_counted_and_left_out():
    rows, _ = _stored_rows()
    for r in rows:
        r["N"] = r["A"]
    rows[0]["N"] = None
    x = sc.score_reader(rows, "N")
    assert x["unparseable_rows"] == [f"{rows[0]['sheet']}:{rows[0]['pos']}"]
    assert sum(x["groups"][k]["judge_vs_reader"]["n"] for k in (1, 2, 3)) == 155


def test_join_refuses_misaligned_rows():
    sample = json.loads((RES / "judgefair_sample.json").read_text())
    stored = json.loads((RES / "judgefair_2026-11-13.json").read_text())
    stored["rows"][0]["pid"] = "zzz"
    with pytest.raises(AssertionError):
        sc.join_stored(sample["rows"], stored["rows"])


def test_load_reader_null_and_bad_length(tmp_path, monkeypatch):
    rows = [dict(sheet=1, pos=1), dict(sheet=1, pos=2)]
    monkeypatch.setattr(sc, "RAW", tmp_path)
    (tmp_path / "x_01.json").write_text(json.dumps([{"grade": "dodge", "coherence": 1}, {"grade": "independent", "coherence": 3}]))
    assert sc.load_reader("x", rows) == [("dodge", 1), ("independent", 3)]
    (tmp_path / "x_01.json").write_text("null")
    assert sc.load_reader("x", rows) == [None, None]
    (tmp_path / "x_01.json").write_text(json.dumps([{"grade": "dodge", "coherence": 1}]))
    with pytest.raises(AssertionError):
        sc.load_reader("x", rows)
