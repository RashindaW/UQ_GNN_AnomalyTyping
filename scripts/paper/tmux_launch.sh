#!/usr/bin/env bash
# Launch a command as a detached, named tmux session with logging + a sentinel.
#
# Usage:
#   bash scripts/paper/tmux_launch.sh <session> <logfile> <sentinel> <cmd...>
#
# Robust against nested-quote issues: writes a small runner script and has tmux
# execute that file (no quote nesting). On completion the command's exit code is
# written to <sentinel>. Monitor with: tmux ls ; tail -f <logfile> ; cat <sentinel>
set -uo pipefail

SESSION="${1:?session name required}"; shift
LOG="${1:?logfile required}"; shift
SENTINEL="${1:?sentinel path required}"; shift
if [ "$#" -lt 1 ]; then echo "no command given" >&2; exit 2; fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$SENTINEL")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launch] session '$SESSION' already exists; not clobbering." >&2
  echo "  attach: tmux attach -t $SESSION   kill: tmux kill-session -t $SESSION" >&2
  exit 3
fi

rm -f "$SENTINEL"
CMD_STR="$*"
RUNNER="$(dirname "$SENTINEL")/.runner_${SESSION}.sh"

# Build the runner file. Unquoted heredoc so $CMD_STR/$LOG/$SENTINEL expand now;
# runtime vars (PIPESTATUS, rc) are backslash-escaped to stay literal.
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -o pipefail
{ $CMD_STR ; } 2>&1 | tee -a '$LOG'
rc=\${PIPESTATUS[0]}
echo "[tmux_launch] exit rc=\$rc" | tee -a '$LOG'
echo \$rc > '$SENTINEL'
EOF
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" "bash '$RUNNER'"
echo "[tmux_launch] launched session=$SESSION"
echo "  runner=$RUNNER"
echo "  log=$LOG"
echo "  sentinel=$SENTINEL"
echo "  attach: tmux attach -t $SESSION"
exit 0
