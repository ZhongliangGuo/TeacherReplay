"""Audit math on synthetic scores (no GPU, no model)."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from teacher_replay import audit as A


# ------------------------------------------------------------------ scores
def test_tr_eq3_single_replay():
    assert A.tr(-1.0, [-1.5, -1.2]) == pytest.approx(0.2)
    assert A.lbar(np.array([-1.0, -2.0], dtype=np.float32)) == pytest.approx(-1.5)
    assert A.lbar([]) is None


def test_tr_m_averages_replays_before_max():
    t = [0.0, 0.0]
    shadows = [[1.0, -1.0], [-1.0, 1.0]]
    # per scorer means first: target 0, shadows 0 and 0 -> 0
    assert A.tr_m(t, shadows) == pytest.approx(0.0)
    # the mean of per-replay TR would be -1: a different statistic
    per_replay = np.mean([A.tr(t[m], [s[m] for s in shadows]) for m in range(2)])
    assert per_replay == pytest.approx(-1.0)
    assert A.tr_m([-1.0], [[-1.5], [-1.2]]) == pytest.approx(A.tr(-1.0, [-1.5, -1.2]))


def test_tr_l_max_of_sums_then_sqrt_n():
    t = np.array([-1.0, -1.0, -1.0, -1.0])
    r1 = np.array([-3.0, 0.0, 0.0, 0.0])  # sum -3
    r2 = np.array([-0.5, -0.5, -0.5, -0.5])  # sum -2
    assert A.tr_l(t, [r1, r2]) == pytest.approx((-4.0 - (-2.0)) / 2.0)
    # not the max of per-token gaps
    assert A.tr_l(t, [r1, r2]) != pytest.approx((t - np.maximum(r1, r2)).sum() / 2.0)


def test_align_same_tokenizer_keeps_common_prefix():
    off = [0, 3, 7, 9]
    al = A.align([off, off[:3]], [[-1, -2, -3, -4], [-5, -6, -7]])
    assert [a.tolist() for a in al] == [[-1, -2, -3], [-5, -6, -7]]


def test_align_word_level_for_different_tokenizers():
    text = "ab cd ef"
    # scorer 1: tokens at 0, 3, 6; scorer 2: tokens at 0, 1, 3, 4 (stops before 'ef')
    al = A.align([[0, 3, 6], [0, 1, 3, 4]], [[-1.0, -2.0, -3.0], [-0.5, -0.5, -1.0, -1.0]], text)
    assert al[0].tolist() == [-1.0, -2.0]
    assert al[1].tolist() == [-1.0, -2.0]
    assert A.align([[0, 3], [0, 1]], [[-1, -1], [-1, -1]], None) is None


def _frame(rows):
    return pd.DataFrame(rows, columns=["id", "m", "lp", "offsets"])


def test_prompt_scores_from_tables():
    off = [0, 2, 4]
    tgt = _frame([("a", 0, [-1.0, -1.0, -1.0], off), ("a", 1, [-2.0, -2.0, -2.0], off), ("b", 0, [], [])])
    s1 = _frame([("a", 0, [-1.5, -1.5, -1.5], off), ("a", 1, [-1.0, -1.0, -1.0], off), ("b", 0, [], [])])
    s2 = _frame([("a", 0, [-1.2, -1.2, -1.2], off), ("a", 1, [-3.0, -3.0, -3.0], off)])
    T = A.ScoredReplays.from_frame(tgt)
    S = [A.ScoredReplays.from_frame(s1), A.ScoredReplays.from_frame(s2)]
    tr = A.prompt_scores("tr", T, S)
    assert list(tr.index) == ["a"]  # b is empty -> undefined
    assert tr["a"] == pytest.approx(-1.0 - (-1.2))
    trm = A.prompt_scores("tr-m", T, S)
    assert trm["a"] == pytest.approx(-1.5 - max(-1.25, -2.1))
    trl = A.prompt_scores("tr-l", T, S)
    assert trl["a"] == pytest.approx((-3.0 - (-3.6)) / math.sqrt(3))


# ------------------------------------------------------------------ test statistic and p-value
def test_standardize_ddof0():
    z = A.standardize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)
    assert A.standardize(np.ones(4)).tolist() == [0.0] * 4


def test_subset_matrix_rows_mark_n_positions():
    rng = np.random.default_rng(0)
    m = A.subset_matrix(rng, 100, 12, 5)
    assert m.shape == (100, 12)
    assert (m.sum(1) == 5).all()


def test_pvalue_formula_matches_direct_count():
    rng = np.random.default_rng(1)
    s, h = rng.normal(0.3, 1, 15), rng.normal(0, 1, 12)
    B, seed = 999, 5
    p = A.owner_test(s, h, B=B, seed=seed)
    z = A.standardize(np.concatenate([s, h]))
    T = z[:15].sum()
    r = np.random.default_rng(np.random.SeedSequence([seed, 7919, 15]))
    Tb = A.subset_matrix(r, B, 27, 15) @ z  # chunking does not change the masks
    assert p == pytest.approx((1 + (Tb >= T - 1e-7).sum()) / (1 + B))
    assert 1 / (B + 1) <= p <= 1


def test_pvalue_invariant_to_affine_score_changes():
    rng = np.random.default_rng(2)
    s, h = rng.normal(0.5, 1, 20), rng.normal(0, 1, 20)
    p = A.owner_test(s, h, B=500, seed=1)
    assert A.owner_test(3 * s + 7, 3 * h + 7, B=500, seed=1) == pytest.approx(p)


def test_pvalue_small_for_strong_shift_and_rejects_nonfinite():
    s, h = np.arange(30) + 100.0, np.arange(30) * 1.0
    assert A.owner_test(s, h, B=999) == pytest.approx(1 / 1000)
    with pytest.raises(ValueError):
        A.owner_test([np.nan, 1.0], [0.0, 1.0])


def test_uniform_pvalues_under_null():
    rng = np.random.default_rng(3)
    O, n, B = 4000, 20, 199
    X = rng.normal(size=(O, 2 * n))
    p = A.shared_pvalues(X[:, :n], X[:, n:], B, [11, 7919, n])
    for a in (0.01, 0.05, 0.1, 0.25, 0.5):
        rate = (p <= a).mean()
        assert rate <= a + 4 * math.sqrt(a * (1 - a) / O)
        assert rate >= a - 4 * math.sqrt(a * (1 - a) / O) - 1 / (B + 1)
    assert p.mean() == pytest.approx(0.5, abs=0.03)


def test_uniform_pvalues_under_null_stratified():
    rng = np.random.default_rng(4)
    O, B = 1500, 99
    ps = []
    for o in range(O):
        ls, lc = rng.integers(0, 3, 12), rng.integers(0, 3, 12)
        # strata shift the scores: the stratified test must stay valid
        vs, vc = rng.normal(size=12) + ls, rng.normal(size=12) + lc
        ps.append(A.strat_pvalue(vs, vc, ls, lc, B, [9, 7919, 12, o]))
    p = np.array(ps)
    for a in (0.05, 0.1, 0.25):
        assert (p <= a).mean() <= a + 4 * math.sqrt(a * (1 - a) / O)


def test_strata_preserve_per_stratum_counts():
    ls, lc = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2, 2])
    f = {0: -2.0, 1: 0.5, 2: 4.0}
    vs, vc = np.array([f[s] for s in ls]), np.array([f[s] for s in lc])
    # scores depend only on the stratum: every count-preserving relabeling gives T_b == T
    assert A.owner_test(vs, vc, B=500, s_strata=ls, h_strata=lc) == 1.0
    assert A.owner_test(vs, vc, B=500) < 1.0


def test_single_stratum_equals_unstratified():
    rng = np.random.default_rng(5)
    s, h = rng.normal(0.2, 1, 25), rng.normal(0, 1, 25)
    ent = [1, 7919, 25]
    p_u = A.shared_pvalues(s[None], h[None], 1200, ent)[0]
    p_s = A.strat_pvalue(s, h, np.zeros(25, int), np.zeros(25, int), 1200, ent)
    assert p_s == pytest.approx(p_u)


def test_log_relabelings():
    assert A.log_relabelings([2, 0, 3]) == pytest.approx(math.log(math.comb(4, 2) * math.comb(6, 3)))


# ------------------------------------------------------------------ Bonferroni, power, n*
def test_bonferroni():
    assert A.bonferroni([0.004, 0.2, 0.5]) == pytest.approx(0.012)
    assert A.bonferroni([0.5, 0.9]) == 1.0
    assert math.isnan(A.bonferroni([0.001, float("nan")]))


def test_n_star_interpolation_and_censoring():
    ns = [10, 20, 50, 100]
    r = A.n_star([0.1, 0.5, 0.9, 1.0], ns)
    w = (0.8 - 0.5) / (0.9 - 0.5)
    assert r.grid == 50 and r.interp == pytest.approx(math.exp(math.log(20) + w * (math.log(50) - math.log(20))))
    assert r.label() == str(round(r.interp))
    low = A.n_star([0.85, 0.9], [10, 20])
    assert low.censored_below and low.label() == "<= 10"
    high = A.n_star([0.1, 0.2], [10, 20])
    assert high.censored_above and math.isinf(high.interp) and high.label() == "> 20"


def test_analytic_n_star():
    assert A.analytic_n_star(1.0, 1.0) == pytest.approx(20.0721, abs=1e-3)
    assert A.analytic_n_star(0.5, 1.0, omega=0.5) > A.analytic_n_star(0.5, 1.0)


def test_simulate_owners_power_and_strata():
    rng = np.random.default_rng(6)
    S, H = rng.normal(1.0, 1, 300), rng.normal(0, 1, 300)
    pv = A.simulate_owners(S, H, [10, 20, 40], n_owners=50, B=199, seed=3)
    ns, fr = A.power_curve(pv, alpha=0.01)
    assert ns == [10, 20, 40] and fr[-1] >= fr[0] and fr[-1] > 0.8
    stS, stH = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    pvs = A.simulate_owners_stratified(S, H, stS, stH, [10, 40], n_owners=20, B=99, seed=1)
    assert np.isfinite(pvs[40]).all()
    # controls can never be matched when H holds no prompt of a stratum that S uses
    pvi = A.simulate_owners_stratified(S, H, np.zeros(300, int), np.ones(300, int), [10], n_owners=5, B=9, seed=1)
    assert np.isnan(pvi[10]).all()
    assert A.power_curve(pvi) == ([], [])


# ------------------------------------------------------------------ per-prompt metrics
def test_auc_matches_pairwise_count_with_ties():
    pos, neg = [3.0, 2.0, 2.0, 5.0], [2.0, 1.0, 4.0]
    pair = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)) / 12
    assert A.auc(pos, neg) == pytest.approx(pair)
    lo, hi = A.bootstrap_auc(pos, neg, B=200)
    assert lo <= hi


def test_tpr_at_fpr():
    neg = np.arange(100.0)
    assert A.tpr_at_fpr([98.5, 99.5, 50.0], neg, 0.01) == pytest.approx(2 / 3)  # threshold = 2nd largest negative (98)
