"""Shadow (and student) training: prompt-masked SFT on teacher responses (paper Sec. III and V-B).

Shadows use exactly the trainer's recipe (same base, teacher, hyper-parameters and seed) and differ
only in their prompts, which come from the same population but never include an owner prompt.
The recipe (paper Sec. VI-A) with TRL's ``SFTTrainer``:

* conversational prompt-completion rows ``{"prompt": [user], "completion": [assistant]}``, no system
  message added (the template's default system turn is kept);
* completion-only loss: prompt tokens condition but carry no loss; the completion labels are the
  response tokens plus the template's ``<|im_end|>`` and the newline after it (no extra EOS);
* loss = token mean over all completion tokens of each optimizer step (TRL/transformers token
  normalization across gradient-accumulation micro-batches);
* one epoch, AdamW (fused), LR 2e-5, linear decay, 3% warmup (``warmup_steps=0.03``), weight decay 0,
  grad-norm clip 1.0, effective batch 128, max length 2048 (keep start; rows left without any
  completion label are dropped), no packing, seed = data seed = 1234;
* single GPU: fp32 master weights with bf16 autocast, gradient checkpointing, SDPA attention with
  the cuDNN SDPA backend disabled;
* the saved checkpoint is bf16 and stops at the template's end-of-turn token.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .chat import chat_end_token
from .io import log_numbers, write_json


@dataclass
class SFTRecipe:
    """Fine-tuning recipe of the paper's students and shadows."""

    epochs: float = 1.0
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    effective_batch: int = 128
    per_device_batch: int = 4
    max_length: int = 2048
    seed: int = 1234
    max_steps: int = -1


def first_responses(replays: pd.DataFrame, drop_finish: Sequence[str] = ()) -> tuple[dict[str, str], dict[str, int]]:
    """``{id: response}`` from a replay table, using the row with the smallest ``m`` per id after
    dropping empty (whitespace-only) responses and rows whose ``finish_reason`` is in ``drop_finish``."""
    df = replays
    c = {"resp_rows": len(df), "resp_empty": 0, "resp_dropped_finish": 0}
    if drop_finish and "finish_reason" in df.columns:
        bad = df["finish_reason"].isin(list(drop_finish))
        c["resp_dropped_finish"] = int(bad.sum())
        df = df[~bad]
    empty = df["response"].isna() | (df["response"].fillna("").astype(str).str.strip() == "")
    c["resp_empty"] = int(empty.sum())
    df = df[~empty].sort_values(["id", "m"], kind="stable").drop_duplicates("id", keep="first")
    return dict(zip(df["id"].astype(str), df["response"].astype(str))), c


def build_examples(prompts: dict[str, str], responses: dict[str, str], ids: Sequence[str], limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Training rows in the order of ``ids`` (repeated ids give repeated rows); ids without a
    response are dropped. ``limit`` keeps the first rows."""
    rows, n_missing = [], 0
    for i in ids:
        t = responses.get(i)
        if t is None:
            n_missing += 1
            continue
        rows.append({"prompt": [{"role": "user", "content": prompts[i]}], "completion": [{"role": "assistant", "content": t}]})
    if limit:
        rows = rows[:limit]
    return rows, {"n_ids": len(ids), "n_no_response": n_missing, "n_examples": len(rows)}


def length_stats(rows: Sequence[dict[str, Any]], tokenizer: Any, max_length: int = 2048) -> dict[str, int]:
    """Token bookkeeping with the template TRL applies (numbers only)."""
    st = {"n_truncated": 0, "n_prompt_overflow_dropped": 0, "n_prefix_mismatch": 0, "total_tokens": 0, "completion_tokens": 0}
    for r in rows:
        p = tokenizer(tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        f = tokenizer(tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=False), add_special_tokens=False)["input_ids"]
        st["n_prefix_mismatch"] += int(f[:len(p)] != p)
        st["total_tokens"] += min(len(f), max_length)
        st["completion_tokens"] += max(0, min(len(f), max_length) - len(p))
        st["n_truncated"] += int(len(f) > max_length)
        st["n_prompt_overflow_dropped"] += int(len(p) >= max_length)
    return st


def check_mask(trainer: Any, tokenizer: Any, rows: Sequence[dict[str, Any]], n: int = 8) -> dict[str, int]:
    """Assert on one collated batch that prompt and padding labels are -100 and that completion
    labels exist and equal the input ids. Raises AssertionError (messages carry no text)."""
    ds = trainer.train_dataset
    n = min(n, len(ds))
    batch = trainer.data_collator([ds[i] for i in range(n)])
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    aligned = len(ds) == len(rows)
    n_comp = 0
    for b in range(n):
        real = int(att[b].sum())
        lab = labels[b]
        assert bool((lab[real:] == -100).all()), "padding positions must be -100"
        if aligned:
            pl = len(tokenizer(tokenizer.apply_chat_template(rows[b]["prompt"], tokenize=False, add_generation_prompt=True),
                               add_special_tokens=False)["input_ids"])
            pl = min(pl, real)
        else:
            nz = (lab[:real] != -100).nonzero()
            pl = int(nz[0]) if len(nz) else real
        assert bool((lab[:pl] == -100).all()), "prompt positions must be -100"
        if real > pl:
            comp = lab[pl:real]
            assert bool((comp != -100).any()), "completion labels must exist"
            assert bool((comp == ids[b][pl:real]).all()), "completion labels must equal input ids"
            n_comp += 1
    assert n_comp > 0, "no row has completion labels"
    return {"mask_rows_checked": n, "mask_rows_with_completion": n_comp}


def apply_end_of_turn(model: Any, tokenizer: Any) -> int | None:
    """Make the template's end-of-turn token (Qwen2.5: ``<|im_end|>``) the model's first eos and
    the tokenizer's eos, keeping the old generation eos ids, so generation stops at turn end."""
    end = chat_end_token(tokenizer)
    if end is None:
        return None
    eid = int(tokenizer.convert_tokens_to_ids(end))
    old = model.generation_config.eos_token_id
    old = [] if old is None else ([old] if isinstance(old, int) else list(old))
    model.generation_config.eos_token_id = [eid] + [i for i in old if i != eid]
    model.config.get_text_config().eos_token_id = eid
    model.config.eos_token_id = eid
    tokenizer.eos_token = end
    return eid


def _save_bf16(trainer: Any, tokenizer: Any, out_dir: str) -> None:
    import torch

    sd = trainer.accelerator.get_state_dict(trainer.model)
    model = trainer.accelerator.unwrap_model(trainer.model)
    if not trainer.accelerator.is_main_process:
        return
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    model.config.dtype = torch.bfloat16
    apply_end_of_turn(model, tokenizer)
    if getattr(model.config.get_text_config(), "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["dtype"] = "bfloat16"
    if "torch_dtype" in cfg:
        cfg["torch_dtype"] = "bfloat16"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _numbers_callback() -> Any:
    from transformers import TrainerCallback

    class NumbersCallback(TrainerCallback):
        """Log numeric training metrics; stop on a non-finite loss or gradient norm."""

        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kw: Any) -> None:
            for k in ("loss", "grad_norm"):
                v = (logs or {}).get(k)
                if isinstance(v, float) and not math.isfinite(v):
                    raise FloatingPointError(f"non-finite {k} at step {state.global_step}")
            if state.is_world_process_zero and logs:
                log_numbers(step=state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

    return NumbersCallback()


def train(
    base: str,
    rows: list[dict[str, Any]],
    out: str,
    recipe: SFTRecipe | None = None,
    check_only: bool = False,
    data_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prompt-masked SFT of ``base`` on ``rows``; writes a bf16 checkpoint to ``out`` (plus
    ``train_stats.json`` and a ``DONE`` marker). ``check_only`` runs the mask check on CPU and stops."""
    recipe = recipe or SFTRecipe()
    if os.path.exists(os.path.join(out, "DONE")) and not check_only:
        raise FileExistsError("output already holds a finished model")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import datasets
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    datasets.disable_progress_bars()
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA gave NaN gradients in our runs
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if recipe.effective_batch % (recipe.per_device_batch * world):
        raise ValueError("effective batch must be divisible by per-device batch x number of processes")
    accum = recipe.effective_batch // (recipe.per_device_batch * world)
    cpu = check_only or not torch.cuda.is_available()
    if not rows:
        raise ValueError("no training examples")

    tok = AutoTokenizer.from_pretrained(base)
    ls = length_stats(rows, tok, recipe.max_length)
    log_numbers(**(data_stats or {}), **ls, per_device_batch=recipe.per_device_batch, grad_accum=accum, processes=world)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if cpu else torch.float32, attn_implementation="sdpa")
    tmp = out.rstrip("/") + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    kw: dict[str, Any] = dict(
        output_dir=tmp, per_device_train_batch_size=recipe.per_device_batch, gradient_accumulation_steps=accum,
        learning_rate=recipe.learning_rate, lr_scheduler_type="linear", warmup_steps=recipe.warmup_ratio,
        num_train_epochs=recipe.epochs, max_steps=recipe.max_steps, bf16=not cpu, use_cpu=cpu,
        seed=recipe.seed, data_seed=recipe.seed, weight_decay=0.0, max_grad_norm=1.0,
        gradient_checkpointing=not cpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=recipe.max_length, packing=False, padding_free=False, completion_only_loss=True, assistant_only_loss=False,
        dataset_num_proc=8 if len(rows) > 5000 else None, dataloader_num_workers=2,
        logging_steps=5, logging_first_step=True, logging_nan_inf_filter=False, save_strategy="no", report_to="none", disable_tqdm=True,
    )
    end = chat_end_token(tok)
    if end is not None and end != tok.eos_token:
        kw["eos_token"] = end
    trainer = SFTTrainer(model=model, args=SFTConfig(**kw), train_dataset=datasets.Dataset.from_list(rows), processing_class=tok)
    for cb in list(trainer.callback_handler.callbacks):
        if type(cb).__name__ in ("PrinterCallback", "ProgressCallback"):
            trainer.remove_callback(cb)
    trainer.add_callback(_numbers_callback())
    ms = check_mask(trainer, tok, rows)
    log_numbers(mask_check_passed=True, **ms)
    if check_only:
        return ms
    t0 = time.time()
    res = trainer.train()
    _save_bf16(trainer, tok, tmp)
    stats = {**(data_stats or {}), **ls, **ms, "train_loss": float(res.training_loss), "global_step": int(trainer.state.global_step),
             "seconds": round(time.time() - t0, 1), "processes": world, "grad_accum": accum, **asdict(recipe)}
    if trainer.accelerator.is_main_process:
        write_json(stats, os.path.join(tmp, "train_stats.json"))
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.replace(tmp, out)
        open(os.path.join(out, "DONE"), "w").close()
    return stats
