"""Table input/output shared by every step.

All files are plain tables with an ``id`` column. Ids are always handled as strings so that
an id written by one step matches the same id read by another. Supported formats, chosen
by file extension: ``.parquet``, ``.jsonl`` and ``.csv``.

Nothing in this module prints table contents: prompts and responses are user text.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence

import pandas as pd


def _fmt(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return "parquet"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"unsupported table extension {ext!r} (use .parquet, .jsonl or .csv)")


def read_table(path: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a table and cast its ``id`` column (if any) to ``str``."""
    fmt = _fmt(path)
    if fmt == "parquet":
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, dtype={"id": str})
    else:
        df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}")
        df = df[list(columns)]
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df


def write_table(df: pd.DataFrame, path: str) -> None:
    """Write a table atomically (temporary file, then rename)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fmt = _fmt(path)
    tmp = path + ".tmp"
    if fmt == "parquet":
        df.to_parquet(tmp, index=False)
    elif fmt == "jsonl":
        df.to_json(tmp, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_prompts(path: str) -> dict[str, str]:
    """``{id: prompt}`` from a table with columns ``id`` and ``prompt``, in file order."""
    df = read_table(path, ["id", "prompt"])
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate prompt ids")
    return dict(zip(df["id"].tolist(), df["prompt"].fillna("").astype(str).tolist()))


def read_ids(path: str) -> list[str]:
    """Ordered id list: a table with an ``id`` column, or a ``.txt`` file with one id per line."""
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return read_table(path, ["id"])["id"].tolist()


def write_json(obj: object, path: str) -> None:
    """Write a JSON file atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def log_numbers(**kw: object) -> None:
    """Print one JSON line of numeric progress information (never text from the data)."""
    keep = {k: v for k, v in kw.items() if isinstance(v, (int, float, bool)) or v is None}
    print(json.dumps(keep, sort_keys=True), flush=True)


def select(prompts: dict[str, str], ids: Iterable[str] | None) -> dict[str, str]:
    """Restrict ``prompts`` to ``ids`` (in the order of ``ids``); all prompts when ``ids`` is None."""
    if ids is None:
        return dict(prompts)
    missing = [i for i in ids if i not in prompts]
    if missing:
        raise KeyError(f"{len(missing)} ids have no prompt")
    return {i: prompts[i] for i in ids}
