"""Tests for theirdir_kappa pure helpers. Shown failing on stubs first."""
import pytest

import theirdir_kappa as K


def test_kappa_perfect_and_chance_and_known_value():
    assert K.cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == pytest.approx(1.0)
    # 2x2: agree 4 of 6 -> po=2/3; marginals 50/50 both -> pe=0.5 -> kappa=1/3
    a = ["A", "A", "A", "B", "B", "B"]
    b = ["A", "A", "B", "B", "B", "A"]
    assert K.cohen_kappa(a, b) == pytest.approx(1 / 3)


def test_kappa_all_one_label_is_defined_not_a_crash():
    assert K.cohen_kappa(["A", "A"], ["A", "A"]) == pytest.approx(1.0)


def test_parse_gemini_list_numbered_lines_and_gaps():
    raw = "1. CORRECT\n2: incorrect\n4) RETRACTED\nnoise"
    assert K.parse_gemini_list(raw, 4) == ["CORRECT", "INCORRECT", "UNCLEAR", "RETRACTED"]
