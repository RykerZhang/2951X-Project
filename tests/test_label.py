"""Hand-built cases for the labeling rules. TODO attribution lane: grow this to at least
10 cases, covering L3_late and other."""
from diagnostics.label import label

OK = {"outcome": "ok", "in_I": True, "in_D": True}


def rec(**kw):
    base = {"in_I": True, "in_I_learned": True, "in_D": True, "outcome": "ok",
            "next_ok_if_stopped_at_T_true": None}
    base.update(kw)
    return base


def test_l4_undetected():
    assert label([OK, rec(in_I=False, in_D=False, outcome="fail_at_start")]) == \
        ("L4", {"pos": 1, "detected": False})


def test_l4_detected():
    assert label([OK, rec(in_I=False, in_I_learned=False, outcome="stuck")])[1]["detected"]


def test_l1():
    assert label([OK, rec(in_I_learned=False, outcome="stuck")])[0] == "L1"


def test_l2():
    assert label([rec(in_D=False, outcome="fail_mid")])[0] == "L2"


def test_l3_early():
    assert label([rec(outcome="wrong_stop")])[0] == "L3_early"


def test_earliest_failure_wins():
    recs = [rec(outcome="wrong_stop"), rec(in_I=False, outcome="fail_at_start")]
    assert label(recs)[0] == "L3_early"
