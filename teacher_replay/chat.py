"""Chat-template helpers shared by replay, scoring and shadow training.

Every step formats a prompt as ONE user turn with the model's own chat template and adds no
system message; whatever default system turn the template inserts is kept. Qwen2.5 base and
Qwen2.5-Instruct insert different default system messages, so a model is always formatted with
its own template unless ``template_tokenizer`` is given (for base models that ship none).
"""

from __future__ import annotations

from typing import Any

END_OF_TURN_CANDIDATES = ("<|im_end|>", "<|eot_id|>", "<|end_of_turn|>")


def chat_prompt(tokenizer: Any, text: str, template_tokenizer: Any = None, enable_thinking: bool | None = None) -> str:
    """Generation prompt for one user turn ``text``.

    Only the template comes from ``template_tokenizer``; tokenization always uses the model's own
    tokenizer. ``enable_thinking`` is passed to the template only when not None (Qwen3 teachers
    and scorers run with ``False``; Qwen2.5 templates ignore it).
    """
    tok = template_tokenizer or tokenizer
    if getattr(tok, "chat_template", None) is None:
        raise ValueError("tokenizer has no chat template; pass a template tokenizer (--template-from)")
    kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kw["enable_thinking"] = bool(enable_thinking)
    return tok.apply_chat_template([{"role": "user", "content": text}], **kw)


def chat_end_token(tokenizer: Any, template_tokenizer: Any = None) -> str | None:
    """End-of-turn token the template writes after an assistant message and the vocabulary holds.

    First of ``<|im_end|>``, ``<|eot_id|>``, ``<|end_of_turn|>`` found in both; None otherwise
    (OLMo-2 templates end turns with the eos token).
    """
    tmpl = getattr(template_tokenizer or tokenizer, "chat_template", None)
    if isinstance(tmpl, dict):
        tmpl = "\n".join(str(v) for v in tmpl.values())
    if not isinstance(tmpl, str):
        return None
    vocab = tokenizer.get_vocab()
    for t in END_OF_TURN_CANDIDATES:
        if t in tmpl and t in vocab:
            return t
    return None


def stop_token_ids(tokenizer: Any, template_tokenizer: Any = None) -> list[int]:
    """Sorted stop ids for sampling: the end-of-turn token (if any) and the tokenizer's eos.

    Qwen2.5 / Qwen3 give ``[151643, 151645]``.
    """
    ids: set[int] = set()
    t = chat_end_token(tokenizer, template_tokenizer)
    if t is not None:
        ids.add(int(tokenizer.convert_tokens_to_ids(t)))
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)


def load_tokenizer(path: str) -> Any:
    """Fast tokenizer of a model (required for offset mappings)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if not getattr(tok, "is_fast", False):
        raise ValueError("a fast tokenizer is required")
    return tok
