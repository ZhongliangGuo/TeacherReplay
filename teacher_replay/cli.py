"""Command-line interface: ``teacher-replay {split,verify,replay,train-shadow,score,audit}``.

Every command prints numbers only (JSON lines); prompts and responses are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import audit as A
from . import owner
from .io import log_numbers, read_ids, read_prompts, read_table, select, write_json, write_table


def _ids(paths: Sequence[str] | None) -> list[str] | None:
    if not paths:
        return None
    out: list[str] = []
    for p in paths:
        out.extend(read_ids(p))
    return out


def _thinking(v: int | None) -> bool | None:
    return None if v is None else bool(v)


# ------------------------------------------------------------------------------------------- split / verify
def cmd_split(a: argparse.Namespace) -> int:
    prompts = select(read_prompts(a.prompts), _ids(a.ids))
    arms = owner.parse_arms(a.arms)
    seed = a.seed if a.seed is not None else owner.new_seed()
    strata = None
    if a.stratify:
        cov = read_table(a.covariates).set_index("id")
        strata = owner.joint_strata([cov.loc[list(prompts), c].to_numpy(dtype=float) for c in a.stratify.split(",")])
    assignment = owner.split_prompts(list(prompts), arms, seed, strata)
    protocol = json.loads(a.protocol) if a.protocol else {}
    salt = owner.new_salt()
    digest = owner.commit(assignment, prompts, salt, protocol)
    df = pd.DataFrame({"id": list(assignment), "arm": list(assignment.values())})
    if strata is not None:
        df["stratum"] = pd.Series(dict(zip(prompts, strata.tolist())))[df["id"]].to_numpy()
    write_table(df, a.out)
    write_json({"commitment": digest, "salt": salt, "seed": seed, "protocol": protocol, "arms": arms}, a.commitment)
    log_numbers(**{f"n_{k}": int(v) for k, v in df["arm"].value_counts().items()})
    print(json.dumps({"commitment": digest}))
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    prompts = read_prompts(a.prompts)
    df = read_table(a.assignment, ["id", "arm"])
    with open(a.commitment) as f:
        rec = json.load(f)
    ok = owner.verify_commitment(rec["commitment"], dict(zip(df["id"], df["arm"])), prompts, rec["salt"], rec.get("protocol"))
    print(json.dumps({"verified": bool(ok)}))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------- replay
def cmd_replay(a: argparse.Namespace) -> int:
    from .replay import Recipe, replay

    prompts = select(read_prompts(a.prompts), _ids(a.ids))
    recipe = Recipe(a.temperature, a.top_p, a.top_k, a.max_tokens, a.repetition_penalty, _thinking(a.enable_thinking))
    man = replay(a.teacher, prompts, a.out, a.m, a.seed, recipe, a.backend, a.template_from, a.hf_batch, a.max_context, a.retries, a.tp)
    log_numbers(**{k: v for k, v in man.items() if not isinstance(v, (dict, list))})
    return 0


# ------------------------------------------------------------------------------------------- train-shadow
def cmd_train(a: argparse.Namespace) -> int:
    from .shadow import SFTRecipe, build_examples, first_responses, train

    prompts = read_prompts(a.prompts)
    responses, c = first_responses(read_table(a.responses, None), tuple(a.drop_finish or ()))
    ids = _ids(a.ids) or list(prompts)
    rows, st = build_examples(prompts, responses, ids, a.limit)
    recipe = SFTRecipe(a.epochs, a.lr, a.warmup_ratio, a.effective_batch, a.per_device_batch, a.max_length, a.seed, a.max_steps)
    train(a.base, rows, a.out, recipe, check_only=a.check_mask, data_stats={**c, **st})
    return 0


# ------------------------------------------------------------------------------------------- score
def cmd_score(a: argparse.Namespace) -> int:
    from .score import score

    prompts = read_prompts(a.prompts)
    replays = read_table(a.replays, ["id", "m", "response"])
    man = score(a.model, prompts, replays, a.out, a.template_from, a.max_response_tokens, a.max_prompt_tokens, _thinking(a.enable_thinking))
    log_numbers(**man)
    return 0


# ------------------------------------------------------------------------------------------- audit
def _candidates(a: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    if a.candidate:
        if a.target or a.shadow:
            raise SystemExit("use either --candidate or --target/--shadow")
        out = []
        for spec in a.candidate:
            name, _, files = spec.partition("=")
            fs = [f for f in files.split(",") if f]
            if len(fs) < 2:
                raise SystemExit("--candidate NAME=TARGET,SHADOW[,SHADOW...]")
            out.append((name, fs[0], fs[1:]))
        return out
    if not a.target or not a.shadow:
        raise SystemExit("--target and at least one --shadow are required")
    return [("candidate", a.target, list(a.shadow))]


def _finite(x: object) -> object:
    """JSON-safe copy: non-finite floats become None, tuples become lists."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _first_replay_covariates(scored: pd.DataFrame) -> pd.DataFrame:
    first = scored.sort_values(["id", "m"], kind="stable").drop_duplicates("id", keep="first").set_index("id")
    return pd.DataFrame({"prompt_tokens": first["n_context_tokens"], "replay_tokens": first["n_tokens_full"]})


def cmd_audit(a: argparse.Namespace) -> int:
    asg = read_table(a.assignment, ["id", "arm"])
    arm = dict(zip(asg["id"], asg["arm"]))
    texts = None
    if a.replays:
        rp = read_table(a.replays, ["id", "m", "response"])
        texts = {(i, int(m)): t for i, m, t in zip(rp["id"], rp["m"], rp["response"])}
    cands = _candidates(a)
    scores, covs = {}, {}
    for name, tgt, shs in cands:
        tdf = read_table(tgt)
        sc = A.prompt_scores(a.method, A.ScoredReplays.from_frame(tdf), [A.ScoredReplays.from_frame(read_table(s)) for s in shs], texts)
        scores[name] = sc
        covs[name] = _first_replay_covariates(tdf)
        if a.covariates:
            extra = read_table(a.covariates).set_index("id")
            covs[name] = covs[name].join(extra, how="outer")
    common = set.intersection(*(set(s.index) for s in scores.values()))
    S = sorted(i for i in common if arm.get(i) == a.suspect_arm)
    H = sorted(i for i in common if arm.get(i) == a.control_arm)
    log_numbers(n_suspect=len(S), n_control=len(H), n_undefined=sum(1 for i in arm if i not in common), n_candidates=len(cands))
    result: dict[str, object] = {"method": a.method, "B": a.B, "seed": a.seed, "alpha": a.alpha, "n_suspect": len(S), "n_control": len(H),
                                 "stratify": a.stratify, "candidates": {}}
    pvals = []
    for name, _, shs in cands:
        sc = scores[name]
        s, h = sc[S].to_numpy(), sc[H].to_numpy()
        ls = lh = None
        if a.stratify:
            cov = covs[name]
            ref = [i for i in arm if i in cov.index]  # all of the owner's prompts fix the cut points
            cols = a.stratify.split(",")
            st = pd.Series(owner.joint_strata([cov.loc[ref, c].to_numpy(dtype=float) for c in cols]), index=ref)
            ls, lh = st[S].to_numpy(), st[H].to_numpy()
        p = A.owner_test(s, h, a.B, a.seed, ls, lh)
        pvals.append(p)
        rec: dict[str, object] = {"p": p, "reject": bool(p < a.alpha), "auc": A.auc(s, h), "auc_ci95": A.bootstrap_auc(s, h),
                                  "mean_diff": float(s.mean() - h.mean()), "smd": A.smd(s, h), "n_shadows": len(shs)}
        if a.power:
            grid = [int(x) for x in a.grid.split(",")] if a.grid else A.PAPER_GRID
            if ls is None:
                pv = A.simulate_owners(s, h, grid, a.n_owners, a.B, A.DEFAULT_SEED + 3)
            else:
                pv = A.simulate_owners_stratified(s, h, ls, lh, grid, a.n_owners, a.B, A.DEFAULT_SEED)
            ns, fr = A.power_curve(pv, a.alpha)
            if ns:
                ns_star = A.n_star(fr, ns)
                rec["power"] = {"n": ns, "frac_reject": fr, "n_star": ns_star.interp, "n_star_label": ns_star.label(),
                                "n_star_analytic": A.analytic_n_star(A.smd(s, h), 1.0, 1.0, a.alpha)}
        result["candidates"][name] = rec  # type: ignore[index]
    result["p"] = A.bonferroni(pvals) if len(pvals) > 1 else pvals[0]
    result["reject"] = bool(result["p"] < a.alpha)  # type: ignore[operator]
    result = _finite(result)
    if a.out:
        write_json(result, a.out)
    if a.scores_out:
        rows = [{"id": i, "arm": arm[i], **{f"score_{n}": float(scores[n][i]) for n in scores}} for i in S + H]
        write_table(pd.DataFrame(rows), a.scores_out)
    print(json.dumps(result, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="teacher-replay", description="Teacher Replay audit of prompt use in distilled language models.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="owner: random split of the prompts and hash commitment (before release)")
    p.add_argument("--prompts", required=True, help="table with columns id, prompt")
    p.add_argument("--ids", nargs="*", help="id files restricting the prompts to the owner's")
    p.add_argument("--arms", default="S=0.5,H=0.5", help="arm fractions, e.g. S=0.4,H=0.4,P=0.2")
    p.add_argument("--seed", type=int, default=None, help="split seed (default: fresh OS randomness, recorded)")
    p.add_argument("--covariates", help="table with id and covariate columns, for a split within strata")
    p.add_argument("--stratify", help="comma-separated covariate columns: split within their joint terciles")
    p.add_argument("--protocol", help="JSON object of audit settings to commit to (score, M, K, strata, B)")
    p.add_argument("--out", required=True, help="assignment table (id, arm[, stratum]); keep private until the audit")
    p.add_argument("--commitment", required=True, help="JSON with digest, salt and seed; publish only the digest")
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("verify", help="check a revealed split against its published commitment")
    p.add_argument("--prompts", required=True)
    p.add_argument("--assignment", required=True)
    p.add_argument("--commitment", required=True)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("replay", help="sample M teacher responses per prompt at the training recipe")
    p.add_argument("--teacher", required=True, help="teacher model id or path")
    p.add_argument("--prompts", required=True)
    p.add_argument("--ids", nargs="*", help="id files restricting the prompts")
    p.add_argument("--out", required=True, help="table: id, m, response, n_tokens, finish_reason")
    p.add_argument("--m", type=int, default=1, help="replays per prompt (paper: M=1; TR-M: 5)")
    p.add_argument("--seed", type=int, default=12, help="replay m uses seed + m")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--enable-thinking", type=int, choices=(0, 1), default=None, help="Qwen3 teachers: 0 (non-thinking)")
    p.add_argument("--template-from", help="tokenizer whose chat template is used (base models without one)")
    p.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    p.add_argument("--hf-batch", type=int, default=4)
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("train-shadow", help="prompt-masked SFT of a shadow (or toy student) with the trainer's recipe")
    p.add_argument("--base", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--responses", required=True, help="teacher responses: id, m, response[, finish_reason]")
    p.add_argument("--ids", nargs="*", help="id files, concatenated in order (default: all prompts)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--effective-batch", type=int, default=128)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    p.add_argument("--drop-finish", nargs="*", help="drop responses with these finish reasons, e.g. length")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--check-mask", action="store_true", help="only check the loss mask (CPU) and exit")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("score", help="teacher-forced per-token log-likelihoods of the replays")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--replays", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--template-from")
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--enable-thinking", type=int, choices=(0, 1), default=None)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("audit", help="TR / TR-L / TR-M scores and the exact owner-level test")
    p.add_argument("--assignment", required=True, help="table: id, arm")
    p.add_argument("--target", help="scored replays under the suspect model")
    p.add_argument("--shadow", action="append", help="scored replays under a shadow (repeat for K shadows)")
    p.add_argument("--candidate", action="append", help="NAME=TARGET,SHADOW[,SHADOW...] per candidate teacher (Bonferroni)")
    p.add_argument("--method", choices=A.METHODS, default="tr")
    p.add_argument("--replays", help="replay table with text; only for TR-L across different tokenizers")
    p.add_argument("--suspect-arm", default="S")
    p.add_argument("--control-arm", default="H")
    p.add_argument("--stratify", help="comma-separated covariates (prompt_tokens, replay_tokens, or --covariates columns)")
    p.add_argument("--covariates", help="table with id and extra covariate columns")
    p.add_argument("--B", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--power", action="store_true", help="also simulate owners and report n*")
    p.add_argument("--grid", help="comma-separated owner sizes for --power")
    p.add_argument("--n-owners", type=int, default=1000)
    p.add_argument("--out", help="write the result JSON here")
    p.add_argument("--scores-out", help="write per-prompt scores (id, arm, score) here")
    p.set_defaults(fn=cmd_audit)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``teacher-replay`` console script."""
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
