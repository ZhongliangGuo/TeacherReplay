"""End-to-end owner split and audit through the CLI on synthetic scored files (no torch)."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from teacher_replay.cli import main


def _scored(path, ids, shift, rng, n_tok=12):
    rows = []
    for i in ids:
        lp = (rng.normal(-1.0, 0.3, n_tok) + shift.get(i, 0.0)).astype(np.float32)
        rows.append({"id": i, "m": 0, "lp": lp, "offsets": np.arange(n_tok, dtype=np.int32) * 3,
                     "n_context_tokens": 20 + int(rng.integers(0, 30)), "n_tokens_full": n_tok + int(rng.integers(0, 40))})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_split_verify_audit(tmp_path, capsys):
    ids = [f"q{i:03d}" for i in range(200)]
    pd.DataFrame({"id": ids, "prompt": [f"text {i}" for i in range(200)]}).to_parquet(tmp_path / "prompts.parquet", index=False)
    assert main(["split", "--prompts", str(tmp_path / "prompts.parquet"), "--arms", "S=0.5,H=0.5", "--seed", "1",
                 "--out", str(tmp_path / "asg.parquet"), "--commitment", str(tmp_path / "commit.json")]) == 0
    assert main(["verify", "--prompts", str(tmp_path / "prompts.parquet"), "--assignment", str(tmp_path / "asg.parquet"),
                 "--commitment", str(tmp_path / "commit.json")]) == 0
    asg = pd.read_parquet(tmp_path / "asg.parquet")
    members = {i: 0.15 for i, a in zip(asg["id"], asg["arm"]) if a == "S"}
    rng = np.random.default_rng(0)
    _scored(tmp_path / "t.parquet", ids, members, rng)
    _scored(tmp_path / "s1.parquet", ids, {}, rng)
    _scored(tmp_path / "s2.parquet", ids, {}, rng)
    capsys.readouterr()
    base = ["audit", "--assignment", str(tmp_path / "asg.parquet"), "--B", "999"]
    assert main(base + ["--target", str(tmp_path / "t.parquet"), "--shadow", str(tmp_path / "s1.parquet"),
                        "--shadow", str(tmp_path / "s2.parquet"), "--out", str(tmp_path / "r.json")]) == 0
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["n_suspect"] == 100 and res["n_control"] == 100
    assert res["p"] < 0.01 and res["reject"]
    for method in ("tr-l", "tr-m"):
        assert main(base + ["--method", method, "--target", str(tmp_path / "t.parquet"), "--shadow", str(tmp_path / "s1.parquet")]) == 0
    assert main(base + ["--stratify", "prompt_tokens,replay_tokens", "--power", "--grid", "10,20,50", "--n-owners", "20",
                        "--target", str(tmp_path / "t.parquet"), "--shadow", str(tmp_path / "s1.parquet"),
                        "--out", str(tmp_path / "rs.json")]) == 0
    rs = json.loads((tmp_path / "rs.json").read_text())
    assert rs["candidates"]["candidate"]["power"]["n"]
    c = f"{tmp_path}/t.parquet,{tmp_path}/s1.parquet,{tmp_path}/s2.parquet"
    assert main(base + ["--candidate", f"a={c}", "--candidate", f"b={c}", "--out", str(tmp_path / "rb.json")]) == 0
    rb = json.loads((tmp_path / "rb.json").read_text())
    pa = rb["candidates"]["a"]["p"]
    assert rb["p"] == min(1.0, 2 * pa)


def test_audit_does_not_import_torch():
    code = "import sys, teacher_replay.audit, teacher_replay.cli; print(int('torch' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    assert out.stdout.strip() == "0"
