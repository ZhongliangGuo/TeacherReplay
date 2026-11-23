#!/usr/bin/env bash
# Toy end-to-end Teacher Replay audit on one 24 GB GPU (about 6-7 minutes on an RTX 3090, HF backend).
#
#   teacher  : Qwen/Qwen2.5-0.5B-Instruct at temperature 0.3, top-p 0.8, top-k 20, repetition
#              penalty 1.1, responses capped at 256 new tokens (same recipe for training data and replays)
#   student  : Qwen/Qwen2.5-0.5B base, prompt-masked SFT on filler + submitted prompts
#   shadows  : A = filler + 200 extra public prompts, B = 600 disjoint public prompts
#   owner    : 400 public alpaca prompts, split 200 submitted (S) / 200 withheld (H)
#
# Toy settings differ from the paper's recipe (CLI defaults) so that a 600-prompt mix of a 0.5B
# student gives a visible signal: teacher temperature 0.3, 2 epochs, LR 5e-5, effective batch 16.
# At the paper's temperature 0.7 this toy scale shows no signal (TR AUC about 0.43-0.50).
# Reference run (RTX 3090): TR p = 1e-4, AUC 0.59; TR-L p = 0.001, AUC 0.58.
# Outputs go to WORK (default <repo>/toy_run, git-ignored). Every command prints numbers only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-python}"
WORK="${WORK:-$ROOT/toy_run}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE="${BASE:-Qwen/Qwen2.5-0.5B}"
GEN=(--temperature "${TEMPERATURE:-0.3}" --max-tokens "${MAX_TOKENS:-256}" --repetition-penalty 1.1 --backend hf --hf-batch "${HF_BATCH:-32}")
SFT=(--epochs "${EPOCHS:-2}" --lr "${LR:-5e-5}" --effective-batch "${EFF_BATCH:-16}" --per-device-batch "${BS:-4}")
TR=("$PY" -m teacher_replay.cli)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
mkdir -p "$WORK"
P="$WORK/prompts.parquet"

echo '{"step": 0}'  # public prompts
"$PY" "$HERE/prepare_alpaca.py" --out-dir "$WORK"

echo '{"step": 1}'  # owner: random split + commitment, before release
"${TR[@]}" split --prompts "$P" --arms S=0.5,H=0.5 --seed 2026 \
  --protocol '{"score": "tr", "M": 1, "K": 2, "B": 10000}' --out "$WORK/assignment.parquet" --commitment "$WORK/commitment.json" \
  --ids "$WORK/owner.txt"
"$PY" - "$WORK" <<'PYEOF'
import sys, pandas as pd
w = sys.argv[1]
d = pd.read_parquet(f"{w}/assignment.parquet")
for arm in ("S", "H"):
    d[d.arm == arm][["id"]].to_parquet(f"{w}/{arm}.parquet", index=False)
PYEOF

echo '{"step": 2}'  # trainer: one teacher draw per training prompt (seed 11), then the suspect student
"${TR[@]}" replay --teacher "$TEACHER" --prompts "$P" --ids "$WORK/S.parquet" "$WORK/filler.txt" "$WORK/shadow_extra.txt" "$WORK/shadow_b.txt" \
  --seed 11 --out "$WORK/train_responses.parquet" "${GEN[@]}"
"${TR[@]}" train-shadow --base "$BASE" --prompts "$P" --responses "$WORK/train_responses.parquet" \
  --ids "$WORK/filler.txt" "$WORK/S.parquet" --out "$WORK/suspect" "${SFT[@]}"

echo '{"step": 3}'  # auditor: K = 2 shadows with the same recipe, on public prompts only
"${TR[@]}" train-shadow --base "$BASE" --prompts "$P" --responses "$WORK/train_responses.parquet" \
  --ids "$WORK/filler.txt" "$WORK/shadow_extra.txt" --out "$WORK/shadow_a" "${SFT[@]}"
"${TR[@]}" train-shadow --base "$BASE" --prompts "$P" --responses "$WORK/train_responses.parquet" \
  --ids "$WORK/shadow_b.txt" --out "$WORK/shadow_b" "${SFT[@]}"

echo '{"step": 4}'  # auditor: fresh replays of the owner's prompts (seed 12), M = 1
"${TR[@]}" replay --teacher "$TEACHER" --prompts "$P" --ids "$WORK/owner.txt" --seed 12 --m 1 --out "$WORK/replays.parquet" "${GEN[@]}"

echo '{"step": 5}'  # auditor: teacher-forced log-likelihoods under the suspect and the shadows
for M in suspect shadow_a shadow_b; do
  "${TR[@]}" score --model "$WORK/$M" --prompts "$P" --replays "$WORK/replays.parquet" --out "$WORK/scored_$M.parquet"
done

echo '{"step": 6}'  # audit: TR and TR-L with the exact owner-level test (B = 10,000)
for METHOD in tr tr-l; do
  "${TR[@]}" audit --assignment "$WORK/assignment.parquet" --method "$METHOD" \
    --target "$WORK/scored_suspect.parquet" --shadow "$WORK/scored_shadow_a.parquet" --shadow "$WORK/scored_shadow_b.parquet" \
    --out "$WORK/audit_$METHOD.json"
done
