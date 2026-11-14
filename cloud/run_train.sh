#!/usr/bin/env bash
# Start (or continue) a training run inside tmux.
#
#   bash cloud/run_train.sh <run_name> [train.py arguments]
#
#   bash cloud/run_train.sh amp_v1 --task G1D-Dance-AMP-v0 --num_envs 4096 --max_iterations 10000
#
# --run_name, --auto_resume and --kit_args --headless are added for you, so running the same command again
# after the machine was interrupted continues from the newest checkpoint. Attach with: tmux attach -t <run_name>
set -euo pipefail

if [ $# -lt 1 ]; then sed -n '2,10p' "$0"; exit 1; fi
NAME="$1"; shift

: "${G1D_HOME:=$HOME/g1d}"
[ -f "$G1D_HOME/env.sh" ] && source "$G1D_HOME/env.sh"
: "${G1D_PYTHON:?run cloud/setup_cloud.sh first}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PROJECT/logs"
OUT="$PROJECT/logs/${NAME}.out"
CMD="cd '$PROJECT' && source '$G1D_HOME/env.sh' && '$G1D_PYTHON' scripts/train.py $* --run_name '$NAME' --auto_resume --kit_args --headless 2>&1 | tee -a '$OUT'"

if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "tmux session '$NAME' is already running: tmux attach -t $NAME"; exit 1
fi
tmux new-session -d -s "$NAME" "bash -c \"$CMD\"; echo '[run_train] process ended, press enter'; read"
echo "started '$NAME' (log: $OUT)"
echo "  watch:        tmux attach -t $NAME     (detach with Ctrl-b d)"
echo "  tensorboard:  $G1D_PYTHON -m tensorboard.main --logdir $PROJECT/logs --port 6006 --bind_all"
