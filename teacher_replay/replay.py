"""Replay: sample M fresh teacher responses per owner prompt at the training recipe (paper Sec. V-A).

The same code also produces the trainer's training responses in the toy example (one draw at a
different seed). Default recipe = the paper's teacher, Qwen3-32B in non-thinking mode: temperature
0.7, top-p 0.8, top-k 20, at most 1,536 new tokens, ``enable_thinking=False``.

Backends
* vLLM (if installed): per-request seeds, so draws do not depend on batching.
* Hugging Face ``generate`` (fallback): batches sorted by prompt length, left padded; torch is
  seeded once per batch with the seed of the batch's first prompt, so draws depend on the batch
  composition and are not bit-equal to vLLM draws (only the recipe is the same).

Repetition penalty: both backends receive ``repetition_penalty`` explicitly (default 1.0, which is
what vLLM uses for Qwen3 teachers, whose generation config sets none). Pass the teacher's own
default when it defines one (Qwen2.5-Instruct: 1.1).

Output table: ``id, m, response, n_tokens, finish_reason``. ``n_tokens`` counts generated tokens
including the final stop token when ``finish_reason == "stop"``; the stored text excludes special
tokens. A JSON manifest next to the output records the recipe and counters (numbers only).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .chat import chat_prompt, load_tokenizer, stop_token_ids
from .io import log_numbers, write_json, write_table


@dataclass
class Recipe:
    """Teacher sampling recipe rho (paper Sec. VI-A defaults)."""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 1536
    repetition_penalty: float = 1.0
    enable_thinking: bool | None = None


def stable_seed(seed: int, id_: str, attempt: int = 0) -> int:
    """Per-prompt 31-bit seed ``blake2b("{seed}:{id}:{attempt}") mod (2^31 - 1) + 1``."""
    h = hashlib.blake2b(f"{seed}:{id_}:{attempt}".encode(), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1) + 1


def is_empty(sample: dict[str, Any]) -> bool:
    """A sample is empty when it has no token ids or only whitespace text."""
    return len(sample["token_ids"]) == 0 or not sample["text"].strip()


class HFBackend:
    """Hugging Face ``generate`` sampler (bf16 on CUDA)."""

    def __init__(self, path: str, stop_ids: Sequence[int], batch_size: int = 4) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            torch.cuda.set_per_process_memory_fraction(min(0.97, max(0.05, (free - 2**30) / total)))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.bs = int(batch_size)
        eos: set[int] = set(int(i) for i in stop_ids)
        for e in (self.model.generation_config.eos_token_id, self.tok.eos_token_id):
            if e is not None:
                eos.update(e if isinstance(e, (list, tuple)) else [e])
        self.eos = sorted(eos)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]

    def generate(self, ids_list: list[list[int]], seeds: list[int], recipe: Recipe) -> list[dict[str, Any]]:
        torch = self.torch
        n = len(ids_list)
        order = sorted(range(n), key=lambda i: len(ids_list[i]))
        res: list[dict[str, Any]] = [{} for _ in range(n)]
        for b0 in range(0, n, self.bs):
            idx = order[b0:b0 + self.bs]
            L = max(len(ids_list[i]) for i in idx)
            inp = torch.full((len(idx), L), self.pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids = ids_list[i]
                inp[r, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
                att[r, L - len(ids):] = 1
            kw: dict[str, Any] = dict(
                input_ids=inp.to(self.device), attention_mask=att.to(self.device), max_new_tokens=int(recipe.max_tokens),
                pad_token_id=self.pad, eos_token_id=list(self.eos), repetition_penalty=float(recipe.repetition_penalty),
            )
            if recipe.temperature > 0:
                kw.update(do_sample=True, temperature=float(recipe.temperature), top_p=float(recipe.top_p), top_k=int(recipe.top_k or 0))
                torch.manual_seed(int(seeds[idx[0]]))
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                seqs = self.model.generate(**kw)[:, L:].cpu()
            for r, i in enumerate(idx):
                gen = seqs[r].tolist()
                cut, fin = min(len(gen), recipe.max_tokens), "length"
                for t, tid in enumerate(gen[:cut]):
                    if tid in self.eos:
                        cut, fin = t + 1, "stop"
                        break
                ids = gen[:cut]
                res[i] = {"text": self.tok.decode(ids, skip_special_tokens=True), "token_ids": ids, "finish_reason": fin}
        return res


class VLLMBackend:
    """vLLM sampler (bf16); prompts are passed as token ids, so vLLM does not re-template them."""

    def __init__(self, path: str, stop_ids: Sequence[int], max_model_len: int, seed: int, tp: int = 1, gpu_mem: float = 0.9) -> None:
        from vllm import LLM

        self.stop_ids = sorted(int(i) for i in stop_ids)
        self.llm = LLM(model=path, tensor_parallel_size=int(tp), gpu_memory_utilization=min(float(gpu_mem), 0.90),
                       seed=int(seed), max_model_len=int(max_model_len), dtype="bfloat16")

    def generate(self, ids_list: list[list[int]], seeds: list[int], recipe: Recipe) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        sps = []
        for i in range(len(ids_list)):
            kw: dict[str, Any] = dict(n=1, temperature=float(recipe.temperature), top_p=float(recipe.top_p),
                                      max_tokens=int(recipe.max_tokens), stop_token_ids=list(self.stop_ids),
                                      repetition_penalty=float(recipe.repetition_penalty))
            if recipe.top_k and recipe.top_k > 0:
                kw["top_k"] = int(recipe.top_k)
            if recipe.temperature > 0:
                kw["seed"] = int(seeds[i])
            sps.append(SamplingParams(**kw))
        outs = self.llm.generate([{"prompt_token_ids": list(map(int, ids))} for ids in ids_list], sps, use_tqdm=False)
        return [{"text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids),
                 "finish_reason": o.outputs[0].finish_reason or "length"} for o in outs]


def generate_retry(backend: Any, ids_list: list[list[int]], seeds: list[int], recipe: Recipe, retries: int = 3) -> tuple[list[dict[str, Any]], int]:
    """Generate, then re-sample empty samples up to ``retries`` rounds (only when sampling) with
    seeds ``seed + 7919 * attempt``; an empty sample is replaced only by a non-empty one."""
    res = backend.generate(ids_list, seeds, recipe)
    retried: set[int] = set()
    if recipe.temperature > 0:
        for attempt in range(1, retries + 1):
            bad = [i for i, s in enumerate(res) if is_empty(s)]
            if not bad:
                break
            sub = backend.generate([ids_list[i] for i in bad], [seeds[i] + 7919 * attempt for i in bad], recipe)
            for b, i in enumerate(bad):
                retried.add(i)
                if not is_empty(sub[b]):
                    res[i] = sub[b]
    return res, len(retried)


def _context_length(path: str) -> int | None:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    for c in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(c, "max_position_embeddings", None) if c is not None else None
        if v:
            return int(v)
    return None


def replay(
    teacher: str,
    prompts: dict[str, str],
    out: str,
    n_replays: int = 1,
    seed: int = 12,
    recipe: Recipe | None = None,
    backend: str = "auto",
    template_from: str | None = None,
    hf_batch: int = 4,
    max_context: int | None = None,
    retries: int = 3,
    tp: int = 1,
) -> dict[str, Any]:
    """Sample ``n_replays`` teacher responses per prompt and write ``id, m, response, n_tokens,
    finish_reason``.

    Replay m (0-based) uses run seed ``seed + m`` and per-prompt seeds ``stable_seed(seed + m, id)``.
    Empty prompts are dropped; so are prompts with ``len(prompt_ids) > context - max_tokens``.
    """
    recipe = recipe or Recipe()
    tok = load_tokenizer(teacher)
    tmpl = load_tokenizer(template_from) if template_from else None
    stop = stop_token_ids(tok, tmpl)
    limits = [c for c in (_context_length(teacher), max_context) if c]
    ctx = min(limits) if limits else None
    counters = {"n_prompts": len(prompts), "n_empty_text": 0, "n_too_long": 0, "n_retried": 0}
    ids, enc = [], []
    for i in sorted(prompts):
        text = prompts[i]
        if not text or not text.strip():
            counters["n_empty_text"] += 1
            continue
        p = tok(chat_prompt(tok, text, tmpl, recipe.enable_thinking), add_special_tokens=False)["input_ids"]
        if ctx is not None and len(p) > ctx - recipe.max_tokens:
            counters["n_too_long"] += 1
            continue
        ids.append(i)
        enc.append(p)
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm"
        except ImportError:
            backend = "hf"
    if backend == "vllm":
        mml = math.ceil(max(256, max((len(p) for p in enc), default=0) + recipe.max_tokens) / 64) * 64
        if ctx is not None:
            mml = min(mml, ctx)
        be: Any = VLLMBackend(teacher, stop, mml, seed, tp=tp)
    else:
        be = HFBackend(teacher, stop, batch_size=hf_batch)
    rows = []
    for m in range(n_replays):
        seeds = [stable_seed(seed + m, i) for i in ids]
        res, n_retry = generate_retry(be, enc, seeds, recipe, retries)
        counters["n_retried"] += n_retry
        for i, s in zip(ids, res):
            rows.append({"id": i, "m": m, "response": s["text"], "n_tokens": len(s["token_ids"]), "finish_reason": s["finish_reason"]})
        log_numbers(replay=m, n_done=len(res), n_stop=sum(s["finish_reason"] == "stop" for s in res))
    df = pd.DataFrame(rows, columns=["id", "m", "response", "n_tokens", "finish_reason"])
    df["m"] = df["m"].astype("int32")
    df["n_tokens"] = df["n_tokens"].astype("int32")
    write_table(df, out)
    manifest = {"backend": backend, "seed": seed, "n_replays": n_replays, "stop_token_ids": stop, "recipe": asdict(recipe),
                "context": ctx, "hf_batch": hf_batch if backend == "hf" else None, **counters, "n_rows": len(df)}
    write_json(manifest, out + ".manifest.json")
    return manifest
