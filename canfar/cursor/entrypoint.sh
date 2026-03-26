#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export RESOLUTION="${VNC_RESOLUTION:-1920x1080}"
export DEPTH="${VNC_COL_DEPTH:-24}"
export NOVNC_PORT="${NOVNC_PORT:-8080}"
export VNC_PORT="${VNC_PORT:-5901}"

Xvfb "$DISPLAY" -screen 0 "${RESOLUTION}x${DEPTH}" -nolisten tcp &
XVFB_PID=$!

fluxbox >/tmp/fluxbox.log 2>&1 &
FLUXBOX_PID=$!

x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -forever -shared -nopw >/tmp/x11vnc.log 2>&1 &
VNC_PID=$!

websockify --web=/usr/share/novnc/ "$NOVNC_PORT" "localhost:$VNC_PORT" >/tmp/websockify.log 2>&1 &
WS_PID=$!

/opt/cursor/Cursor.AppImage --no-sandbox --disable-gpu --user-data-dir=/home/cursor/.cursor-data >/tmp/cursor.log 2>&1 &
CURSOR_PID=$!

cleanup() {
  kill "$CURSOR_PID" "$WS_PID" "$VNC_PID" "$FLUXBOX_PID" "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$CURSOR_PID"
