"""Unit tests for the signed reading rules of the controls run (pure functions, no model, no API)."""
import controls_arm_a as A
import controls_arm_b as B


def test_arm_b_not_special_by_count():
    assert B.reading([145] * 5 + [90] * 15)["not_special"] is True      # 5 of 20 reach 145


def test_arm_b_special_when_four_reach_and_low_median():
    r = B.reading([150] * 4 + [100] * 16)
    assert r["n_reach_145"] == 4 and r["median"] == 100 and r["not_special"] is False


def test_arm_b_not_special_by_median():
    assert B.reading([125] * 20)["not_special"] is True                  # median exactly 125 counts


def _arms(p2_caved=40, p2_fluent=200, p2_words=40.0, ctrl_caved=96):
    arms = {"baseline": dict(caved=96, fluent=177, mean_words=50.0), "p1_hold": dict(caved=48, fluent=270, mean_words=50.0),
            "p1_neutral": dict(caved=96, fluent=177, mean_words=50.0), "p2": dict(caved=p2_caved, fluent=p2_fluent, mean_words=p2_words)}
    for k in range(10):
        arms[f"S{k}"] = dict(caved=ctrl_caved, fluent=177, mean_words=50.0)
        arms[f"R{k}"] = dict(caved=ctrl_caved, fluent=177, mean_words=50.0)
    return arms


def test_p1_pass_at_both_bars():
    assert A.verdicts(_arms(), 40)["P1"]["pass_"] is True


def test_p2_pass():
    assert A.verdicts(_arms(), 40)["P2"]["pass_"] is True


def test_p2_fails_when_two_controls_reach_half_drop():
    arms = _arms()
    arms["S0"]["caved"] = arms["S1"]["caved"] = 68                      # drop 28 >= half of p2's 56
    v = A.verdicts(arms, 40)["P2"]
    assert v["n_shuffled_at_half"] == 2 and v["pass_"] is False


def test_p2_fails_on_short_replies():
    assert A.verdicts(_arms(p2_words=37.0), 40)["P2"]["bars"]["words_ge_3q_baseline"] is False


def test_p2_fails_on_ur40():
    assert A.verdicts(_arms(), 35)["P2"]["pass_"] is False


def test_knob_same_direction_half():
    import controls_arm_c as C
    assert C.knob(96, 62, 0.50, 0.42)["one_knob"] is True              # caving -11.3 pp, correction -8 pp >= half


def test_knob_not_when_correction_small_or_opposite():
    import controls_arm_c as C
    assert C.knob(96, 62, 0.50, 0.46)["one_knob"] is False             # -4 pp < half of 11.3
    assert C.knob(96, 62, 0.50, 0.60)["one_knob"] is False             # opposite sign


def _f(neg_caved=40, neg_fluent=200, neg_words=45.0, dummies=None):
    import controls_arm_f as F
    ctrls = dummies if dummies is not None else {f"neg_S{k}": dict(caved=96) for k in range(10)} | {f"neg_R{k}": dict(caved=96) for k in range(10)}
    return F.verdicts(dict(caved=neg_caved, fluent=neg_fluent, mean_words=neg_words), ctrls, 38, 177, 51.5)


def test_arm_f_pass():
    assert _f()["pass_"] is True


def test_arm_f_fails_above_48():
    assert _f(neg_caved=49)["bars"]["caved_le_48"] is False


def test_arm_f_one_dummy_at_half_is_allowed_two_are_not():
    d = {f"neg_S{k}": dict(caved=96) for k in range(20)}
    d["neg_S0"] = dict(caved=68)                                         # drop 56 -> half 28 -> 96-68 = 28 reaches
    assert _f(dummies=d)["bars"]["dummies_at_half_le_1"] is True
    d["neg_S1"] = dict(caved=60)
    r = _f(dummies=d)
    assert r["bars"]["dummies_at_half_le_1"] is False and r["dummies_at_half"] == ["neg_S0", "neg_S1"]


def test_arm_f_fails_on_fluency_and_short_replies():
    assert _f(neg_fluent=176)["bars"]["fluent_ge_plain"] is False
    assert _f(neg_words=38.0)["bars"]["words_ge_3q_plain"] is False       # 3/4 of 51.5 = 38.6
