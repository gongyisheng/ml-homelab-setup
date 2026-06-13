#!/bin/bash
# Install the GPU idle-alert as a cron job (idempotent). Re-running updates the entry.
#
#   bash setup_crontab.sh            # every 15 min (default)
#   CRON_SCHEDULE="*/30 * * * *" bash setup_crontab.sh
#   bash setup_crontab.sh --remove   # uninstall
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALERT="$SCRIPT_DIR/gpu_idle_alert.sh"
LOG="/tmp/gpu_idle_alert.log"
MARKER="# ml-homelab gpu_idle_alert"
SCHEDULE="${CRON_SCHEDULE:-*/15 * * * *}"

current="$(crontab -l 2>/dev/null || true)"
# Drop any existing managed line.
filtered="$(printf '%s\n' "$current" | grep -vF "$MARKER" || true)"

if [ "${1:-}" = "--remove" ]; then
    printf '%s\n' "$filtered" | crontab -
    echo "Removed gpu_idle_alert cron job."
    exit 0
fi

chmod +x "$ALERT"
entry="$SCHEDULE $ALERT >> $LOG 2>&1 $MARKER"
printf '%s\n%s\n' "$filtered" "$entry" | grep -v '^$' | crontab -
echo "Installed: $entry"
echo "Logs: $LOG"
