"""Audit: Teacher Replay scores and the exact owner-level randomization test (paper Sec. V-B and V-C).

Pure numpy and pandas; this module never imports torch.

Orientation: a larger score means more member-like. ``S`` are the owner's submitted prompts,
``H`` the withheld controls (``P`` an optional placebo arm that no model saw).

Scores (from teacher-forced per-token log-likelihoods, see :mod:`teacher_replay.score`)

* :func:`tr` -- Eq. 3 with M = 1: ``lbar_target(y'_1|x) - max_k lbar_shadow_k(y'_1|x)``.
* :func:`tr_m` -- Eq. 3 with M replays (TR-M): average the replays per scorer first, then take
  the max over shadows.
* :func:`tr_l` -- TR-L: ``(sum_i t_i - max_k sum_i r_{k,i}) / sqrt(n)`` on aligned tokens.

Test (Eq. 4, Algorithm 1, Proposition 4)

* :func:`owner_test` -- z-score the pooled scores of one owner (ddof 0), ``T = sum_{x in S} z(x)``,
  B uniform relabelings (within strata if any), ``p = (1 + #{T_b >= T}) / (1 + B)``.
* :func:`bonferroni` -- one test per candidate teacher, ``p = min(1, |C| min_c p_c)``.

Power (Definition 1, Proposition 6)

* :func:`simulate_owners`, :func:`simulate_owners_stratified`, :func:`n_star`,
  :func:`analytic_n_star`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

TOL = 1e-7
"""Absolute tolerance on the z scale in ``T_b >= T - TOL`` (guards float summation order)."""
UNSTRAT_CHUNK = 250
"""Relabelings drawn per block in the unstratified test (does not change the masks)."""
STRAT_CHUNK = 500
"""Relabelings drawn per block in the stratified test (part of the stratified RNG stream)."""
RELABEL_TAG = 7919
"""Constant mixed into every relabeling SeedSequence."""
MATCH_TAG = 31
"""Constant mixed into the SeedSequence of the stratified control matching."""
DEFAULT_SEED = 20261024
"""Base seed of the paper's owner simulations (power uses DEFAULT_SEED + 3, placebo + 4)."""
PAPER_GRID = sorted(set(range(10, 101, 10)) | {50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000})
"""Owner sizes n on which the paper interpolates n*."""

_WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------------------- per-prompt scores
def lbar(lp: Sequence[float]) -> float | None:
    """Mean token log-likelihood (Eq. 1) in float64; None for an empty response."""
    x = np.asarray(lp, dtype=np.float64)
    return float(x.mean()) if len(x) else None


def tr(target: float, shadows: Sequence[float]) -> float:
    """TR, Eq. 3 with M = 1: ``lbar_target - max_k lbar_k`` (K = len(shadows) >= 1)."""
    if not len(shadows):
        raise ValueError("TR needs at least one shadow")
    return float(target - max(shadows))


def tr_m(target: Sequence[float], shadows: Sequence[Sequence[float]]) -> float:
    """TR-M, Eq. 3: ``mean_m lbar_target(y'_m) - max_k mean_m lbar_k(y'_m)``.

    ``target`` has one mean log-likelihood per replay (M,), ``shadows`` one row per shadow (K, M).
    Replays are averaged per scorer FIRST and the max over shadows is taken after, so this is
    not the mean over replays of per-replay TR.
    """
    t = np.asarray(target, dtype=np.float64)
    r = np.asarray(shadows, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != t.shape[0] or r.shape[0] == 0:
        raise ValueError("shadows must have shape (K, M) with M = len(target)")
    return float(t.mean() - r.mean(axis=1).max())


def tr_l(target: Sequence[float], shadows: Sequence[Sequence[float]]) -> float:
    """TR-L: ``(sum_i t_i - max_k sum_i r_{k,i}) / sqrt(n)`` (paper Sec. V-B, length-aware variant).

    Inputs are ALIGNED per-unit log-likelihoods of the first replay (see :func:`align`): the max
    is over per-shadow sums and the division by sqrt(n) comes once, after the max.
    """
    t = np.asarray(target, dtype=np.float64)
    r = np.asarray(shadows, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != len(t) or r.shape[0] == 0 or len(t) == 0:
        raise ValueError("TR-L needs K >= 1 aligned arrays of the same non-zero length")
    return float((t.sum() - r.sum(axis=1).max()) / math.sqrt(len(t)))


def word_starts(text: str) -> np.ndarray:
    """Character starts of the whitespace-delimited words (regex ``\\S+``) of ``text``."""
    return np.fromiter((m.start() for m in _WORD_RE.finditer(text)), dtype=np.int64)


def word_sum(starts: np.ndarray, offsets: Sequence[int], values: Sequence[float]) -> np.ndarray:
    """Sum of per-token values within each word; tokens before the first word go to word 0."""
    nw = len(starts)
    if nw == 0:
        return np.zeros(0)
    w = np.clip(np.searchsorted(starts, np.asarray(offsets), side="right") - 1, 0, nw - 1)
    return np.bincount(w, weights=np.asarray(values, dtype=np.float64), minlength=nw)


def align(
    offsets: Sequence[Sequence[int]],
    values: Sequence[Sequence[float]],
    text: str | None = None,
) -> list[np.ndarray] | None:
    """Align the per-token log-likelihoods of several scorers on one response.

    Case 1 (same tokenizer, unit "token"): when every scorer's first ``n = min len(offsets)``
    token offsets agree, all arrays are cut to that common prefix.
    Case 2 (different tokenizers, unit "word"): with the response ``text``, keep the words that
    start before ``lim = min_k (last token offset_k + 1)``, drop tokens at or after ``lim``, and sum
    each scorer's values per word. Returns None when no unit remains or the text is missing.
    """
    offs = [np.asarray(o, dtype=np.int64) for o in offsets]
    vals = [np.asarray(v, dtype=np.float64) for v in values]
    n = min(len(o) for o in offs)
    if n and all(len(v) >= n and np.array_equal(o[:n], offs[0][:n]) for o, v in zip(offs, vals)):
        return [v[:n] for v in vals]
    if text is None or n == 0:
        return None
    lim = min(int(o[-1]) + 1 for o in offs)
    st = word_starts(text)
    st = st[st < lim]
    if len(st) == 0:
        return None
    return [word_sum(st, o[o < lim], v[: len(o)][o < lim]) for o, v in zip(offs, vals)]


@dataclass
class ScoredReplays:
    """Per-token log-likelihoods of one scorer: ``rows[id][m] = (lp, offsets)``."""

    rows: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> ScoredReplays:
        """Build from a scored table (columns ``id``, ``m``, ``lp``, ``offsets``)."""
        rows: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for i, m, lp, off in zip(df["id"].astype(str), df["m"], df["lp"], df["offsets"]):
            rows.setdefault(i, {})[int(m)] = (np.asarray(lp, dtype=np.float32), np.asarray(off, dtype=np.int64))
        return cls(rows)

    def lbar(self, id_: str, m: int) -> float | None:
        r = self.rows.get(id_, {}).get(m)
        return None if r is None else lbar(r[0])


METHODS = ("tr", "tr-l", "tr-m")


def prompt_scores(
    method: str,
    target: ScoredReplays,
    shadows: Sequence[ScoredReplays],
    texts: Mapping[tuple[str, int], str] | None = None,
) -> pd.Series:
    """Per-prompt score of every id the target scored (index id, float64 values).

    ``tr`` and ``tr-l`` use the first replay (lowest ``m`` of the target); ``tr-m`` uses every
    replay ``m`` the target scored. A prompt whose score is undefined (a scorer lacks the replay,
    or a replay is empty) is left out; this does not depend on the assignment.
    ``texts[(id, m)]`` (replay text) is only needed by ``tr-l`` with different tokenizers.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not shadows:
        raise ValueError("at least one shadow is required")
    out: dict[str, float] = {}
    for id_, reps in target.rows.items():
        ms = sorted(reps)
        if method == "tr-m":
            t = [target.lbar(id_, m) for m in ms]
            r = [[s.lbar(id_, m) for m in ms] for s in shadows]
            if any(v is None for v in t) or any(v is None for row in r for v in row):
                continue
            out[id_] = tr_m(t, r)  # type: ignore[arg-type]
            continue
        m0 = ms[0]
        if method == "tr":
            t0 = target.lbar(id_, m0)
            r0 = [s.lbar(id_, m0) for s in shadows]
            if t0 is None or any(v is None for v in r0):
                continue
            out[id_] = tr(t0, r0)  # type: ignore[arg-type]
            continue
        recs = [target.rows[id_][m0]] + [s.rows.get(id_, {}).get(m0) for s in shadows]
        if any(r is None or len(r[0]) == 0 for r in recs):
            continue
        al = align([r[1] for r in recs], [r[0] for r in recs], None if texts is None else texts.get((id_, m0)))  # type: ignore[index]
        if al is None or len(al[0]) == 0:
            continue
        out[id_] = tr_l(al[0], al[1:])
    return pd.Series(out, dtype=np.float64)


# --------------------------------------------------------------------------------------- randomization test
def standardize(z: np.ndarray) -> np.ndarray:
    """Row-wise z-score over the pooled values (np.std, ddof = 0); a zero sd divides by 1."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=-1, keepdims=True)
    sd = z.std(axis=-1, keepdims=True)
    return z / np.where(sd > 0, sd, 1.0)


def subset_matrix(rng: np.random.Generator, c: int, N: int, n: int) -> np.ndarray:
    """``c x N`` 0/1 float64 matrix; each row marks a uniformly random n-subset of N positions.

    Keys are ``rng.random((c, N))`` filled row by row; a row marks its n smallest keys.
    """
    if n <= 0:
        return np.zeros((c, N))
    keys = rng.random((c, N))
    kth = np.partition(keys, n - 1, axis=1)[:, n - 1:n]
    return (keys <= kth).astype(np.float64)


def _require_finite(*arrays: np.ndarray) -> None:
    for a in arrays:
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            raise ValueError("non-finite scores passed to the randomization test")


def shared_pvalues(
    Xs: np.ndarray, Xc: np.ndarray, B: int, entropy: Sequence[int], chunk: int = UNSTRAT_CHUNK
) -> np.ndarray:
    """Unstratified test of many owners at once, with the SAME B relabelings for every owner.

    ``Xs`` (O, nS) suspect scores, ``Xc`` (O, nH) control scores. Each owner row is z-scored over
    its pooled ``nS + nH`` values (suspects first), ``T = sum`` of the suspects' z, and every
    relabeling marks a uniform nS-subset of the pooled positions, drawn from
    ``default_rng(SeedSequence(entropy))``. Returns p (O,) = (1 + #{T_b >= T - TOL}) / (1 + B).
    """
    Xs = np.atleast_2d(np.asarray(Xs, dtype=np.float64))
    Xc = np.atleast_2d(np.asarray(Xc, dtype=np.float64))
    _require_finite(Xs, Xc)
    nS = Xs.shape[1]
    Z = standardize(np.concatenate([Xs, Xc], axis=1))
    N = Z.shape[1]
    obs = Z[:, :nS].sum(axis=1)
    cnt = np.zeros(Z.shape[0])
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    done = 0
    while done < B:
        c = min(chunk, B - done)
        sums = subset_matrix(rng, c, N, nS) @ Z.T
        cnt += (sums >= (obs - TOL)[None, :]).sum(axis=0)
        done += c
    return (1.0 + cnt) / (1.0 + B)


def strat_pvalue(
    vs: np.ndarray,
    vc: np.ndarray,
    ls: np.ndarray,
    lc: np.ndarray,
    B: int,
    entropy: Sequence[int],
    chunk: int = STRAT_CHUNK,
) -> float:
    """Stratified test of one owner (Algorithm 1 "within strata").

    ``vs`` / ``vc`` suspect / control scores with stratum labels ``ls`` / ``lc``. The pooled values
    (suspects first) are z-scored together (strata do not enter the standardization) and
    ``T = sum`` of the suspects' z. Each relabeling picks, independently in every stratum, a
    uniform a_s-subset of its a_s + b_s pooled positions (a_s suspects, b_s controls), strata in
    ascending label order, in blocks of ``chunk`` relabelings from
    ``default_rng(SeedSequence(entropy))``. This preserves the per-stratum suspect counts.
    """
    vs = np.asarray(vs, dtype=np.float64)
    vc = np.asarray(vc, dtype=np.float64)
    _require_finite(vs, vc)
    n = len(vs)
    z = standardize(np.concatenate([vs, vc]))
    obs = z[:n].sum()
    lab = np.concatenate([np.asarray(ls), np.asarray(lc)])
    is_sus = np.arange(len(lab)) < n
    segs = []
    for s in np.unique(lab):
        idx = np.flatnonzero(lab == s)
        segs.append((z[idx], int(is_sus[idx].sum())))
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    cnt = 0
    done = 0
    while done < B:
        c = min(chunk, B - done)
        sums = np.zeros(c)
        for zs, a in segs:
            if a == len(zs):
                sums += zs.sum()
            elif a > 0:
                sums += subset_matrix(rng, c, len(zs), a) @ zs
        cnt += int((sums >= obs - TOL).sum())
        done += c
    return (1.0 + cnt) / (1.0 + B)


def owner_test(
    s_scores: Sequence[float],
    h_scores: Sequence[float],
    B: int = 10_000,
    seed: int = 0,
    s_strata: Sequence[int] | None = None,
    h_strata: Sequence[int] | None = None,
) -> float:
    """Exact owner-level p-value, Eq. 4 (Algorithm 1; valid by Proposition 4).

    Unstratified: relabelings from ``SeedSequence([seed, 7919, n_S])``. Stratified (labels for both
    arms): ``SeedSequence([seed, 7919, n_S, 0])``. Reject H0 at level alpha iff ``p < alpha``.
    """
    s = np.asarray(s_scores, dtype=np.float64)
    h = np.asarray(h_scores, dtype=np.float64)
    if len(s) == 0 or len(h) == 0:
        raise ValueError("both arms need at least one score")
    if (s_strata is None) != (h_strata is None):
        raise ValueError("give strata for both arms or for neither")
    if s_strata is None:
        return float(shared_pvalues(s[None, :], h[None, :], B, [seed, RELABEL_TAG, len(s)])[0])
    return strat_pvalue(s, h, np.asarray(s_strata), np.asarray(h_strata), B, [seed, RELABEL_TAG, len(s), 0])


def bonferroni(pvalues: Sequence[float]) -> float:
    """Bonferroni over candidate teachers/recipes: ``min(1, |C| * min_c p_c)``; NaN propagates.

    Valid whatever the dependence between the candidate tests (paper Sec. V-C).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if len(p) == 0:
        raise ValueError("no p-values")
    if np.isnan(p).any():
        return float("nan")
    return float(min(1.0, len(p) * p.min()))


def log_relabelings(counts: Sequence[int]) -> float:
    """log of the number of distinct matched stratified relabelings ``prod_s C(2 m_s, m_s)``.

    A level alpha with ``log(alpha) < -log_relabelings`` cannot be reached.
    """
    m = np.asarray(counts, dtype=float)
    m = m[m > 0]
    return float(sum(math.lgamma(2 * x + 1) - 2 * math.lgamma(x + 1) for x in m))


# --------------------------------------------------------------------------------------- simulated owners
def owner_draws(NS: int, NH: int, n_max: int, n_owners: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Index draws of simulated owners: all S permutations first, then all H permutations, from
    ``default_rng(seed)``; owner o at size n uses the first n of each (sizes are nested)."""
    rng = np.random.default_rng(seed)
    n_max = min(n_max, NS, NH)
    iS = np.stack([rng.permutation(NS)[:n_max] for _ in range(n_owners)])
    iH = np.stack([rng.permutation(NH)[:n_max] for _ in range(n_owners)])
    return iS, iH


def simulate_owners(
    S: Sequence[float],
    H: Sequence[float],
    grid: Sequence[int] = PAPER_GRID,
    n_owners: int = 1000,
    B: int = 10_000,
    seed: int = DEFAULT_SEED + 3,
) -> dict[int, np.ndarray]:
    """Unstratified p-values of ``n_owners`` simulated owners with n suspects from S and n
    controls from H, for every n of ``grid`` with n <= min(|S|, |H|).

    Owners come from :func:`owner_draws` (``seed``); relabelings from
    ``SeedSequence([seed, 7919, n])``, shared by all owners. Returns ``{n: p (n_owners,)}``.
    Placebo check: pass the placebo arm as S with ``seed = DEFAULT_SEED + 4`` and n = 500.
    """
    S = np.asarray(S, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    _require_finite(S, H)
    sizes = [int(n) for n in grid if n <= min(len(S), len(H))]
    if not sizes:
        return {}
    iS, iH = owner_draws(len(S), len(H), max(sizes), n_owners, seed)
    return {n: shared_pvalues(S[iS[:, :n]], H[iH[:, :n]], B, [seed, RELABEL_TAG, n]) for n in sizes}


def matching_orders(stH: np.ndarray, n_strata: int, n_owners: int, seed: int, purpose: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-owner control matching order: for each stratum s ascending, a permutation of the H rows
    in s from ``default_rng(SeedSequence([seed, 31, purpose, o]))``. Returns (perm, segment offsets)."""
    rows = np.arange(len(stH))
    Hs = [rows[stH == s] for s in range(n_strata)]
    seg = np.concatenate([[0], np.cumsum([len(h) for h in Hs])]).astype(int)
    out = np.empty((n_owners, len(rows)), dtype=np.int64)
    for o in range(n_owners):
        rng = np.random.default_rng(np.random.SeedSequence([seed, MATCH_TAG, purpose, o]))
        out[o] = np.concatenate([rng.permutation(h) for h in Hs]) if len(rows) else np.zeros(0, dtype=np.int64)
    return out, seg


def simulate_owners_stratified(
    S: Sequence[float],
    H: Sequence[float],
    stS: Sequence[int],
    stH: Sequence[int],
    grid: Sequence[int] = PAPER_GRID,
    n_owners: int = 1000,
    B: int = 10_000,
    seed: int = DEFAULT_SEED,
    purpose: int = 0,
) -> dict[int, np.ndarray]:
    """Stratified (quoted) power design: suspects are the unstratified S prefixes of
    :func:`owner_draws` (``seed + 3``); controls are matched count for count within strata (first
    m_s rows of the owner's :func:`matching_orders` of each stratum); the test is
    :func:`strat_pvalue` with ``SeedSequence([seed, 7919, n, o])``.

    An owner without enough controls in some stratum is infeasible (p = NaN).
    Returns ``{n: p (n_owners,)}``.
    """
    S = np.asarray(S, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    stS = np.asarray(stS, dtype=np.int64)
    stH = np.asarray(stH, dtype=np.int64)
    _require_finite(S, H)
    sizes = [int(n) for n in grid if n <= min(len(S), len(H))]
    if not sizes:
        return {}
    n_strata = int(max(stS.max(), stH.max())) + 1
    rng = np.random.default_rng(seed + 3)
    PS = np.stack([rng.permutation(len(S)) for _ in range(n_owners)])
    perm, seg = matching_orders(stH, n_strata, n_owners, seed, purpose)
    out = {}
    for n in sizes:
        p = np.full(n_owners, np.nan)
        for o in range(n_owners):
            sus = PS[o, :n]
            cnt = np.bincount(stS[sus], minlength=n_strata)
            ctl = []
            for s in np.flatnonzero(cnt):
                rows = perm[o, seg[s]:seg[s + 1]]
                if len(rows) < cnt[s]:
                    ctl = None
                    break
                ctl.append(rows[:cnt[s]])
            if ctl is None:
                continue
            c = np.concatenate(ctl)
            p[o] = strat_pvalue(S[sus], H[c], stS[sus], stH[c], B, [seed, RELABEL_TAG, n, o])
        out[n] = p
    return out


def power_curve(pvalues: Mapping[int, np.ndarray], alpha: float = 0.01, infeasible_max: float = 0.01) -> tuple[list[int], list[float]]:
    """Fraction of feasible owners with ``p < alpha`` per n. The grid ends before the first n at
    which more than ``infeasible_max`` of the owners are infeasible (NaN)."""
    ns, fr = [], []
    for n in sorted(pvalues):
        p = np.asarray(pvalues[n])
        ok = np.isfinite(p)
        if (1.0 - ok.mean()) > infeasible_max or not ok.any():
            break
        ns.append(int(n))
        fr.append(float((p[ok] < alpha).mean()))
    return ns, fr


@dataclass
class NStar:
    """n*: first grid n with power >= target, its log-linear interpolation, and censoring."""

    grid: float
    interp: float
    censored_below: bool
    censored_above: bool

    def label(self) -> str:
        """Printable value: ``<= n`` / ``> n_last`` when censored, else the rounded interpolation."""
        if self.censored_below:
            return f"<= {int(self.grid)}"
        if self.censored_above:
            return f"> {int(self.grid)}"
        return str(int(round(self.interp)))


def n_star(fracs: Sequence[float], ns: Sequence[int], target: float = 0.8) -> NStar:
    """Smallest number of submitted prompts at which ``target`` of owners reject (Definition 1).

    Interpolation between the bracketing grid points (n0, f0), (n1, f1):
    ``w = (target - f0) / (f1 - f0)`` (1 if f1 <= f0), ``n* = exp(log n0 + w (log n1 - log n0))``.
    Reached at the first n: censored below (n* <= that n). Never reached: n* = inf, censored above
    (``grid`` then holds the last n).
    """
    ns, fr = list(ns), list(fracs)
    if not ns:
        raise ValueError("empty grid")
    for i, (n, f) in enumerate(zip(ns, fr)):
        if f >= target:
            if i == 0:
                return NStar(float(n), float(n), True, False)
            n0, f0 = ns[i - 1], fr[i - 1]
            w = (target - f0) / (f - f0) if f > f0 else 1.0
            return NStar(float(n), float(math.exp(math.log(n0) + w * (math.log(n) - math.log(n0)))), False, False)
    return NStar(float(ns[-1]), float("inf"), False, True)


def smd(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference ``(mean a - mean b) / pooled SD`` (ddof = 1)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sd = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else float("nan")


def analytic_n_star(delta: float, sigma: float, omega: float = 1.0, alpha: float = 0.01, power: float = 0.8) -> float:
    """Eq. 5 (Proposition 6): ``(z_{1-alpha} + z_{power})^2 (2 sigma^2 + omega (1 - omega) delta^2) / (omega delta)^2``.

    With ``sigma = 1`` and ``delta = smd(S, H)`` this is ``2 (z_.99 + z_.80)^2 / (omega d)^2`` at omega = 1.
    """
    z = NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(power)
    return float(z * z * (2.0 * sigma * sigma + omega * (1.0 - omega) * delta * delta) / (omega * delta) ** 2)


# --------------------------------------------------------------------------------------- per-prompt metrics
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (as scipy.stats.rankdata(method="average"))."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Per-prompt AUC between members (pos) and non-members (neg): Mann-Whitney U / (n1 n0)."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    r = _avg_ranks(np.concatenate([p, n]))
    n1, n0 = len(p), len(n)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auc(pos: Sequence[float], neg: Sequence[float], B: int = 2000, seed: int = DEFAULT_SEED + 1) -> tuple[float, float]:
    """95% percentile bootstrap interval of :func:`auc` (resample each arm with replacement;
    draw b uses ``rng.integers(0, n1, n1)`` then ``rng.integers(0, n0, n0)``)."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        ip = rng.integers(0, len(p), len(p))
        ineg = rng.integers(0, len(n), len(n))
        vals[b] = auc(p[ip], n[ineg])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def tpr_at_fpr(pos: Sequence[float], neg: Sequence[float], fpr: float = 0.01) -> float:
    """TPR at a fixed FPR: threshold = the ``floor(fpr * N)``-th largest negative (0-based)."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.sort(np.asarray(neg, dtype=np.float64))[::-1]
    m = int(math.floor(fpr * len(n) + 1e-9))
    thr = n[m] if m < len(n) else -np.inf
    return float((p > thr).mean())
