"""Score: teacher-forced per-token log-likelihoods of replays under a model (Eq. 1).

For each (prompt x, replay y') the model sees ``chat_prompt(x) + y'`` and every response token
y_t is scored by ``log p(y_t | x, y_<t)``. Exact conventions (they define the paper's numbers):

* The prompt string and the response text are tokenized SEPARATELY, both with
  ``add_special_tokens=False``; the response also with ``split_special_tokens=True``. Joint
  tokenization would change the ids at the boundary.
* No end-of-turn or EOS token is appended or scored: only the response's own text tokens.
* The response keeps its FIRST ``max_response_tokens`` tokens (default 512); a prompt longer than
  ``max_prompt_tokens`` (2048) keeps its LAST 2048 tokens.
* bf16 model, SDPA attention, right padding with ``pad_token_id`` (else eos). Hidden states come
  from the decoder only (``use_cache=False``); the last hidden state is cast to fp32 and multiplied
  by the fp32 ``lm_head`` weight (TF32 matmul allowed), followed by a FULL-vocabulary fp32
  log-softmax over the padded lm_head rows. ``model(...).logits`` in bf16 gives other numbers.
* Items are sorted by length and batched greedily (at most 64 sequences and 16,384 padded tokens);
  bf16 kernels make results depend slightly on batch composition (about 1e-3 nats).

Output table: ``id, m, lp (list<float32>), offsets, offsets_end (list<int32>, character span of
each scored token in the response), n_tokens (scored), n_tokens_full, n_context_tokens, truncated``,
one row per input row, in input order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .chat import chat_prompt, load_tokenizer
from .io import log_numbers, write_json, write_table


def _encode(tok: Any, texts: Sequence[str], split_special: bool = False, chunk: int = 2048) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    ids: list[list[int]] = []
    offs: list[list[tuple[int, int]]] = []
    kw: dict[str, Any] = dict(add_special_tokens=False, return_offsets_mapping=True)
    if split_special:
        kw["split_special_tokens"] = True
    for s in range(0, len(texts), chunk):
        enc = tok(list(texts[s:s + chunk]), **kw)
        ids.extend(enc["input_ids"])
        offs.extend(enc["offset_mapping"])
    return ids, offs


def build_items(
    tok: Any,
    prompts: dict[str, str],
    rows: pd.DataFrame,
    template_tok: Any = None,
    max_response_tokens: int = 512,
    max_prompt_tokens: int = 2048,
    enable_thinking: bool | None = None,
) -> tuple[list[tuple[list[int], int, int]], list[dict[str, Any]], dict[str, int]]:
    """Token sequences to score: ``(prompt_ids + response_ids, start, end)`` per replay row.

    ``rows`` has columns ``id, m, response``. Rows whose id has no prompt are skipped and counted.
    """
    counters = {"missing_prompt": 0, "empty_response": 0, "truncated": 0, "prompt_truncated": 0}
    keep = rows[rows["id"].isin(prompts.keys())].reset_index(drop=True)
    counters["missing_prompt"] = int(len(rows) - len(keep))
    need = sorted(set(keep["id"]))
    pids, _ = _encode(tok, [chat_prompt(tok, prompts[i], template_tok, enable_thinking) for i in need])
    pcache = dict(zip(need, pids))
    rids, roffs = _encode(tok, [r if isinstance(r, str) else "" for r in keep["response"]], split_special=True)
    items, meta = [], []
    for id_, m, ri, ro in zip(keep["id"], keep["m"], rids, roffs):
        n_full = len(ri)
        trunc = n_full > max_response_tokens
        ri, ro = ri[:max_response_tokens], ro[:max_response_tokens]
        counters["truncated"] += int(trunc)
        counters["empty_response"] += int(n_full == 0)
        p = pcache[id_]
        if len(p) > max_prompt_tokens:
            p = p[-max_prompt_tokens:]
            counters["prompt_truncated"] += 1
        items.append((list(p) + list(ri), len(p), len(p) + len(ri)))
        meta.append({"id": id_, "m": int(m), "n_full": n_full, "n_prompt": len(p), "truncated": trunc,
                     "offs": np.asarray([o[0] for o in ro], dtype=np.int32), "ends": np.asarray([o[1] for o in ro], dtype=np.int32)})
    return items, meta, counters


class _Head:
    """fp32 log-softmax of the lm_head applied to bf16 hidden states, in row chunks."""

    def __init__(self, model: Any, chunk_elems: int = 2**27) -> None:
        emb = model.get_output_embeddings()
        self.w = emb.weight.detach().float()
        b = getattr(emb, "bias", None)
        self.b = None if b is None else b.detach().float()
        self.cap = getattr(model.config, "final_logit_softcapping", None)
        self.rows = max(16, int(chunk_elems // self.w.shape[0]))

    def __call__(self, h: Any, targets: Any) -> Any:
        import torch

        out = torch.empty(h.shape[0], dtype=torch.float32, device=h.device)
        prev = torch.backends.cuda.matmul.allow_tf32
        if h.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            for s in range(0, h.shape[0], self.rows):
                e = min(h.shape[0], s + self.rows)
                a = h[s:e].float() @ self.w.T
                if self.b is not None:
                    a += self.b
                if self.cap:
                    a = self.cap * torch.tanh(a / self.cap)
                a -= torch.logsumexp(a, dim=-1, keepdim=True)
                out[s:e] = a.gather(1, targets[s:e, None])[:, 0]
                del a
        finally:
            if h.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = prev
        return out


def score_items(
    model: Any,
    items: Sequence[tuple[list[int], int, int]],
    pad_id: int,
    max_batch_tokens: int = 16384,
    max_batch_size: int = 64,
) -> list[np.ndarray]:
    """Per-token log-likelihoods ``lp`` (float32) of ``ids[start:end]`` for every item, in input order.

    Position t is predicted from the hidden state at t - 1. Empty items (start == end) are not run.
    On CUDA out-of-memory a batch is halved recursively.
    """
    import torch

    device = next(model.parameters()).device
    base = getattr(model, getattr(model, "base_model_prefix", ""), None)
    if base is None or base is model:
        base = model.get_decoder()
    head = _Head(model)
    results: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in items]
    order = sorted((i for i, it in enumerate(items) if it[2] > it[1]), key=lambda i: (-len(items[i][0]), i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) >= max_batch_size or (len(cur) + 1) * len(items[cur[0]][0]) > max_batch_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    def run(batch: list[int]) -> dict[int, np.ndarray]:
        try:
            with torch.inference_mode():
                lens = [len(items[i][0]) for i in batch]
                ids = torch.full((len(batch), max(lens)), pad_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                for r, i in enumerate(batch):
                    ids[r, :lens[r]] = torch.as_tensor(items[i][0], dtype=torch.long)
                    att[r, :lens[r]] = 1
                ids, att = ids.to(device), att.to(device)
                hid = base(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                spans = [(items[i][1], items[i][2]) for i in batch]
                h = torch.cat([hid[r, s - 1:e - 1] for r, (s, e) in enumerate(spans)], 0)
                t = torch.cat([ids[r, s:e] for r, (s, e) in enumerate(spans)], 0)
                del hid
                lp = head(h, t).cpu().numpy()
            out, pos = {}, 0
            for i, (s, e) in zip(batch, spans):
                out[i] = lp[pos:pos + e - s]
                pos += e - s
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return {**run(batch[:mid]), **run(batch[mid:])}

    for bi, batch in enumerate(batches):
        for i, lp in run(batch).items():
            results[i] = lp
        if (bi + 1) % 50 == 0:
            log_numbers(batches_done=bi + 1, batches=len(batches))
    return results


def load_scorer(path: str) -> tuple[Any, Any, int]:
    """bf16 model with SDPA attention on the GPU (CPU fp32 otherwise), its fast tokenizer, pad id."""
    import torch
    from transformers import AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if cuda else torch.float32, attn_implementation="sdpa")
    model = model.to("cuda" if cuda else "cpu").eval()
    tok = load_tokenizer(path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    return model, tok, int(pad)


def score(
    model_path: str,
    prompts: dict[str, str],
    replays: pd.DataFrame,
    out: str,
    template_from: str | None = None,
    max_response_tokens: int = 512,
    max_prompt_tokens: int = 2048,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Score every replay row (``id, m, response``) under ``model_path`` and write the scored table."""
    model, tok, pad = load_scorer(model_path)
    tmpl = load_tokenizer(template_from) if template_from else None
    items, meta, counters = build_items(tok, prompts, replays, tmpl, max_response_tokens, max_prompt_tokens, enable_thinking)
    lps = score_items(model, items, pad)
    df = pd.DataFrame({
        "id": [r["id"] for r in meta],
        "m": np.asarray([r["m"] for r in meta], dtype=np.int32),
        "lp": [np.asarray(x, dtype=np.float32) for x in lps],
        "offsets": [r["offs"] for r in meta],
        "offsets_end": [r["ends"] for r in meta],
        "n_tokens": np.asarray([len(x) for x in lps], dtype=np.int32),
        "n_tokens_full": np.asarray([r["n_full"] for r in meta], dtype=np.int32),
        "n_context_tokens": np.asarray([r["n_prompt"] for r in meta], dtype=np.int32),
        "truncated": np.asarray([r["truncated"] for r in meta], dtype=bool),
    })
    write_table(df, out)
    manifest = {"n_rows": len(df), "n_scored_tokens": int(df["n_tokens"].sum()), "vocab": int(model.get_output_embeddings().weight.shape[0]),
                "max_response_tokens": max_response_tokens, "max_prompt_tokens": max_prompt_tokens,
                "enable_thinking": -1 if enable_thinking is None else int(enable_thinking), "pad_id": pad, **counters}
    write_json(manifest, out + ".manifest.json")
    return manifest
