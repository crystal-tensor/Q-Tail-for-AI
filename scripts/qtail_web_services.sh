#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
SERVE="$ROOT/node_modules/.bin/serve"
PAGE_PATH="/qtail-droid-full-training"
PAGE_MARKER="Q-Tail DROID Full Evidence"

if /sbin/mount | /usr/bin/grep -Fq " on /Volumes/ORICO ("; then
  LOG="/Volumes/ORICO/qtail_full_training/logs/qtail-web-services.log"
else
  LOG="$ROOT/.tmp/qtail-web-services.log"
fi
mkdir -p "$(/usr/bin/dirname "$LOG")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

port_is_listening() {
  /usr/sbin/lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

listener_pid() {
  /usr/sbin/lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null \
    | head -1
}

service_owned() {
  local port="$1"
  local pid command expected

  pid="$(listener_pid "$port")"
  [ -n "$pid" ] || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  expected="node $SERVE -l tcp://0.0.0.0:$port"
  [ "$command" = "$expected" ]
}

content_healthy() {
  local port="$1"

  /usr/bin/curl -fsS \
    --connect-timeout 2 \
    --max-time 5 \
    "http://127.0.0.1:$port$PAGE_PATH" 2>/dev/null \
    | /usr/bin/grep -Fq "$PAGE_MARKER"
}

service_healthy() {
  local port="$1"

  port_is_listening "$port" \
    && service_owned "$port" \
    && content_healthy "$port"
}

ensure_service() {
  local port="$1"
  local session="$2"

  if service_healthy "$port"; then
    log "$session healthy on port $port"
    return 0
  fi

  if port_is_listening "$port" && ! service_owned "$port"; then
    log "refusing to stop foreign listener on port $port"
    return 1
  fi

  /usr/bin/screen -S "$session" -X quit >/dev/null 2>&1 || true
  /usr/bin/screen -wipe >/dev/null 2>&1 || true
  for _ in {1..10}; do
    if ! port_is_listening "$port"; then
      break
    fi
    sleep 1
  done
  if port_is_listening "$port"; then
    log "owned listener on port $port did not stop"
    return 1
  fi

  /usr/bin/screen -dmS "$session" /bin/zsh -lc \
    "cd '$ROOT' && exec '$SERVE' -l tcp://0.0.0.0:$port"

  for _ in {1..20}; do
    if service_healthy "$port"; then
      log "started $session on port $port"
      return 0
    fi
    sleep 1
  done

  log "failed to start $session on port $port"
  return 1
}

if [ ! -x "$SERVE" ]; then
  log "serve binary missing: $SERVE"
  exit 1
fi

ensure_service 54655 qtail-web-54655
ensure_service 6222 qtail-web-6222
