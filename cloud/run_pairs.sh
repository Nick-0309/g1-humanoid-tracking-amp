#!/usr/bin/env bash
# Paired ablation for the landing rewards: same initialisation, same motion, same iterations,
# only --foot_rewards differs, so the difference is attributable to the reward alone.
#
#   tmux new -s pairs 'bash cloud/run_pairs.sh'
#   ITERS=3000 ENVS=4096 bash cloud/run_pairs.sh
#   TRACK_CKPT=logs/g1d_tracking/<run>/model_9999.pt bash cloud/run_pairs.sh
#   MOTION=motions/dance_102_ground.npz bash cloud/run_pairs.sh   # the faithful variant of the cleanup
#   VARIANTS="land_clean land_ctrl land_orig" bash cloud/run_pairs.sh
#   DRY=1 bash cloud/run_pairs.sh            # print the commands, run nothing
#
# land_clean  cleaned reference + landing rewards          (the treatment)
# land_ctrl   cleaned reference, no landing rewards        (the control)
# land_orig   original reference + landing rewards         (optional: isolates the data cleanup)
#
# Every run continues from its own newest checkpoint when the script is started again (--auto_resume), so a
# machine restart only costs the unfinished part of one run. EVAL after each run reads the motion that
# train.py saved in the run's params/ directory.
set -euo pipefail

ENVS="${ENVS:-4096}"
ITERS="${ITERS:-3000}"
MOTION="${MOTION:-motions/dance_102_clean.npz}"
ORIGINAL_MOTION="${ORIGINAL_MOTION:-motions/dance_102.npz}"
VARIANTS="${VARIANTS:-land_clean land_ctrl}"
DRY="${DRY:-0}"

: "${G1D_HOME:=$HOME/g1d}"
[ -f "$G1D_HOME/env.sh" ] && source "$G1D_HOME/env.sh"
: "${G1D_PYTHON:?point G1D_PYTHON at the Isaac Lab interpreter}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

TRACK="G1D-Dance-Tracking-v0"
# highest iteration index across all tracking runs (a smoke run with model_5 never wins over model_9999)
TRACK_CKPT="${TRACK_CKPT:-$(ls -1 "$PROJECT"/logs/g1d_tracking/*/model_*.pt 2>/dev/null \
    | awk -F'/' '{n=$NF; sub(/^model_/, "", n); sub(/\.pt$/, "", n); print n"\t"$0}' \
    | sort -n | tail -n 1 | cut -f2- || true)}"
if [ -z "$TRACK_CKPT" ]; then
  echo "no tracking checkpoint under logs/g1d_tracking/*/model_*.pt; train the baseline first (see README)"
  exit 1
fi
if [ ! -f "$MOTION" ]; then
  echo "motion '$MOTION' does not exist; copy it from the local machine (see README)"
  exit 1
fi
case " $VARIANTS " in *" land_orig "*)
  if [ ! -f "$ORIGINAL_MOTION" ]; then
    echo "motion '$ORIGINAL_MOTION' does not exist (needed by land_orig)"
    exit 1
  fi ;;
esac

echo "motion:     $MOTION"
echo "original:   $ORIGINAL_MOTION"
echo "init_from:  $TRACK_CKPT"
echo "runs:       $VARIANTS   ($ITERS iterations, $ENVS envs each)"

train() {  # name, motion, extra train.py arguments
  local name="$1"
  local motion="$2"
  shift 2
  local -a cmd
  cmd=("$G1D_PYTHON" scripts/train.py --task "$TRACK" --num_envs "$ENVS" --motion "$motion"
       --init_from "$TRACK_CKPT" --max_iterations "$ITERS" "$@"
       --run_name "$name" --auto_resume --kit_args --headless)
  echo
  echo "=== $name ($(basename "$motion")) ${*}"
  echo "+ ${cmd[*]}"
  if [ "$DRY" = "1" ]; then
    return 0
  fi
  "${cmd[@]}"
  # eval without --motion: the policy loader picks up the copy saved next to the checkpoint
  "$G1D_PYTHON" scripts/eval.py --task "$TRACK" --load_run "$name" --num_envs 64 --kit_args --headless
}

for name in $VARIANTS; do
  case "$name" in
    land_clean) train land_clean "$MOTION" --foot_rewards ;;
    land_ctrl) train land_ctrl "$MOTION" ;;
    land_orig) train land_orig "$ORIGINAL_MOTION" --foot_rewards ;;
    *) echo "unknown variant '$name'"; exit 1 ;;
  esac
done

if [ "$DRY" = "1" ]; then
  exit 0
fi

"$G1D_PYTHON" scripts/summarize_runs.py logs --out logs/landing_summary.md
echo
echo "done. Compare in logs/landing_summary.md: touchdowns (policy / ref), contact mismatch, swing contact,"
echo "and Metrics/motion/error_body_pos. Acceptance: mismatch <= 12 %, swing contact <= 30 %,"
echo "touchdowns 14-26 per foot, error_body_pos <= 0.030 m ."
