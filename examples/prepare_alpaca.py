"""Toy data: public prompts from the cached ``yahma/alpaca-cleaned`` dataset.

A prompt is the ``instruction`` field, followed by a blank line and the ``input`` field when that is
non-empty. Prompts are de-duplicated, a seeded random sample is drawn, and the sample is cut into
four disjoint id lists:

* ``owner.txt``         -- the data owner's prompts (split later into S / H by ``teacher-replay split``)
* ``filler.txt``        -- the public part of the trainer's mix
* ``shadow_extra.txt``  -- extra public prompts for shadow A (shadow A = filler + extra)
* ``shadow_b.txt``      -- prompts for shadow B, disjoint from the trainer's mix

Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--owner", type=int, default=400)
    ap.add_argument("--filler", type=int, default=400)
    ap.add_argument("--shadow-extra", type=int, default=200)
    ap.add_argument("--shadow-b", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("yahma/alpaca-cleaned", "alpaca_data_cleaned.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    seen, prompts = set(), []
    for i, r in enumerate(data):
        ins, inp = (r.get("instruction") or "").strip(), (r.get("input") or "").strip()
        text = ins + ("\n\n" + inp if inp else "")
        if ins and text not in seen:
            seen.add(text)
            prompts.append((f"alpaca-{i:05d}", text))
    sizes = {"owner": a.owner, "filler": a.filler, "shadow_extra": a.shadow_extra, "shadow_b": a.shadow_b}
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(prompts), size=sum(sizes.values()), replace=False)
    os.makedirs(a.out_dir, exist_ok=True)
    chosen = [prompts[i] for i in pick]
    pd.DataFrame(chosen, columns=["id", "prompt"]).to_parquet(os.path.join(a.out_dir, "prompts.parquet"), index=False)
    pos = 0
    for name, n in sizes.items():
        with open(os.path.join(a.out_dir, f"{name}.txt"), "w") as f:
            f.write("".join(pid + "\n" for pid, _ in chosen[pos:pos + n]))
        pos += n
    print(json.dumps({"n_pool": len(prompts), **{f"n_{k}": v for k, v in sizes.items()}}))


if __name__ == "__main__":
    main()
