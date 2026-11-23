"""Owner split, commitment and strata."""

import numpy as np
import pytest

from teacher_replay import owner


def _prompts(n=101):
    return {f"p{i}": f"prompt number {i}" for i in range(n)}


def test_split_is_deterministic_and_order_free():
    ids = list(_prompts())
    arms = owner.parse_arms("S=0.4,H=0.4,P=0.2")
    a = owner.split_prompts(ids, arms, seed=7)
    b = owner.split_prompts(list(reversed(ids)), arms, seed=7)
    assert a == b
    assert owner.split_prompts(ids, arms, seed=8) != a
    counts = {k: sum(v == k for v in a.values()) for k in arms}
    assert counts == {"S": 40, "H": 41, "P": 20}  # round(0.4 * 101) = 40, round(0.8 * 101) = 81


def test_split_within_strata_keeps_fractions_per_stratum():
    ids = [f"x{i}" for i in range(300)]
    strata = np.arange(300) % 3
    a = owner.split_prompts(ids, {"S": 0.5, "H": 0.5}, seed=1, strata=strata)
    lab = dict(zip(ids, strata))
    for s in range(3):
        n_s = sum(1 for i in ids if lab[i] == s and a[i] == "S")
        assert n_s == 50


def test_parse_arms_rejects_bad_fractions():
    with pytest.raises(ValueError):
        owner.parse_arms("S=0.5,H=0.4")
    with pytest.raises(ValueError):
        owner.parse_arms("S=0.5,S=0.5")


def test_commitment_binds_split_prompts_salt_protocol():
    prompts = _prompts(20)
    asg = owner.split_prompts(list(prompts), {"S": 0.5, "H": 0.5}, seed=3)
    proto = {"score": "tr", "M": 1, "K": 2, "B": 10000}
    d = owner.commit(asg, prompts, "aa" * 32, proto)
    assert len(d) == 64
    assert d == owner.commit(dict(reversed(list(asg.items()))), prompts, "aa" * 32, proto)
    assert owner.verify_commitment(d, asg, prompts, "aa" * 32, proto)
    flipped = dict(asg)
    k = next(iter(flipped))
    flipped[k] = "H" if flipped[k] == "S" else "S"
    assert not owner.verify_commitment(d, flipped, prompts, "aa" * 32, proto)
    assert not owner.verify_commitment(d, asg, {**prompts, k: "changed"}, "aa" * 32, proto)
    assert not owner.verify_commitment(d, asg, prompts, "bb" * 32, proto)
    assert not owner.verify_commitment(d, asg, prompts, "aa" * 32, {**proto, "B": 999})


def test_terciles_and_joint_strata():
    x = np.arange(1, 10, dtype=float)
    c = owner.tercile_cuts(x)
    assert c == pytest.approx((np.quantile(x, 1 / 3), np.quantile(x, 2 / 3)))
    t = owner.tercile(x, c)
    assert t.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert owner.tercile([c[0]], c)[0] == 0 and owner.tercile([c[1]], c)[0] == 1
    st = owner.joint_strata([x, x[::-1]])
    assert st.tolist() == [t0 * 3 + t1 for t0, t1 in zip(t, t[::-1])]
    with pytest.raises(ValueError):
        owner.joint_strata([np.array([1.0, np.nan, 2.0])])
