"""Owner side: random split, hash commitment, strata (paper Sec. III, "Data owner and audit").

Before the prompts leave the owner's control, a uniformly random subset S is submitted and the
rest is withheld as controls H (optionally a placebo arm P). A SHA-256 commitment to the split,
published at release, lets a third party check later that S and H were not chosen after the
fact (Proposition 4 needs the split to be random and fixed before any score is computed).

Strata (paper Sec. VI-A): joint terciles of covariates such as prompt length and replay length.
Tercile cut points come from a reference set that does not depend on the assignment, e.g. all of
the owner's prompts.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence

import numpy as np

COMMITMENT_VERSION = 1


def parse_arms(spec: str) -> dict[str, float]:
    """Parse ``"S=0.5,H=0.5"`` into ordered ``{arm: fraction}`` (fractions must sum to 1)."""
    arms: dict[str, float] = {}
    for part in spec.split(","):
        name, _, frac = part.partition("=")
        name = name.strip()
        if not name or name in arms:
            raise ValueError("arm names must be non-empty and distinct")
        arms[name] = float(frac)
    total = sum(arms.values())
    if any(f < 0 for f in arms.values()) or abs(total - 1.0) > 1e-9:
        raise ValueError("arm fractions must be non-negative and sum to 1")
    return arms


def _allocate(rng: np.random.Generator, rows: np.ndarray, arms: Mapping[str, float], out: np.ndarray) -> None:
    """Uniformly random allocation of ``rows`` to the arms; arm a gets the slice between
    round(n * cumulative fraction) boundaries of one random permutation."""
    n = len(rows)
    perm = rows[rng.permutation(n)]
    bounds = np.rint(np.cumsum([0.0] + list(arms.values())) * n).astype(int)
    bounds[-1] = n
    for a, name in enumerate(arms):
        out[perm[bounds[a]:bounds[a + 1]]] = name


def split_prompts(
    ids: Sequence[str],
    arms: Mapping[str, float],
    seed: int,
    strata: Sequence[int] | None = None,
) -> dict[str, str]:
    """Uniformly random assignment of prompt ids to arms (e.g. S / H / P).

    Ids are sorted first, so the result depends only on the id set, the fractions and the seed.
    With ``strata`` (one integer label per id, aligned with ``ids``) the allocation is made
    independently within every stratum, in ascending stratum order, which is the design the
    stratified test of :func:`teacher_replay.audit.owner_test` assumes.
    """
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
    sid = [ids[i] for i in order]
    rng = np.random.default_rng(seed)
    out = np.empty(len(sid), dtype=object)
    if strata is None:
        _allocate(rng, np.arange(len(sid)), arms, out)
    else:
        lab = np.asarray(strata)[order]
        for s in np.unique(lab):
            _allocate(rng, np.flatnonzero(lab == s), arms, out)
    return dict(zip(sid, out.tolist()))


def new_salt() -> str:
    """32 random bytes (hex) that keep the commitment hiding until the owner reveals it."""
    return secrets.token_hex(32)


def new_seed() -> int:
    """A fresh 63-bit split seed from the operating system's randomness."""
    return secrets.randbits(63)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def commitment_payload(
    assignment: Mapping[str, str],
    prompts: Mapping[str, str],
    salt: str,
    protocol: Mapping[str, object] | None = None,
) -> bytes:
    """Canonical bytes that are hashed: salt, audit protocol (score, M, K, strata, B, ...), and
    one ``[id, arm, sha256(prompt)]`` triple per prompt sorted by id."""
    rows = [[i, assignment[i], _text_hash(prompts[i])] for i in sorted(assignment)]
    obj = {"version": COMMITMENT_VERSION, "salt": salt, "protocol": dict(protocol or {}), "assignment": rows}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def commit(
    assignment: Mapping[str, str],
    prompts: Mapping[str, str],
    salt: str,
    protocol: Mapping[str, object] | None = None,
) -> str:
    """SHA-256 commitment (hex) to the split. Publish it at release; keep salt and split private."""
    return hashlib.sha256(commitment_payload(assignment, prompts, salt, protocol)).hexdigest()


def verify_commitment(
    digest: str,
    assignment: Mapping[str, str],
    prompts: Mapping[str, str],
    salt: str,
    protocol: Mapping[str, object] | None = None,
) -> bool:
    """True iff the revealed split, prompts, salt and protocol reproduce the published digest."""
    return commit(assignment, prompts, salt, protocol) == digest


def tercile_cuts(reference: Sequence[float]) -> tuple[float, float]:
    """Cut points ``np.quantile(x, [1/3, 2/3])`` (linear) over the finite reference values."""
    x = np.asarray(reference, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite reference values for tercile cut points")
    c = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return float(c[0]), float(c[1])


def tercile(x: Sequence[float], cuts: tuple[float, float]) -> np.ndarray:
    """Tercile index: 0 if x <= c1, 1 if c1 < x <= c2, else 2."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= cuts[0], 0, np.where(x <= cuts[1], 1, 2)).astype(np.int64)


def joint_strata(covariates: Sequence[Sequence[float]], reference: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    """Joint tercile stratum of every prompt: ``((t0 * 3) + t1) * 3 + t2 ...`` (3^d cells).

    ``covariates[d]`` holds covariate d for every prompt; ``reference[d]`` the values that fix the
    cut points (default: the covariates themselves, which is assignment-independent when they cover
    all of the owner's prompts). A missing (non-finite) covariate is an error.
    """
    if not covariates:
        raise ValueError("at least one covariate is needed")
    ref = covariates if reference is None else reference
    code = np.zeros(len(covariates[0]), dtype=np.int64)
    for x, r in zip(covariates, ref):
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("missing covariate value; strata are undefined")
        code = code * 3 + tercile(x, tercile_cuts(r))
    return code
