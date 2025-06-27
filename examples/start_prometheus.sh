#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
if command -v "realpath" &>/dev/null; then
    # Best to ensure absolute paths for Docker's bind mounts.
    SCRIPT_DIR=$(realpath "$SCRIPT_DIR")
fi

docker run --rm --name prometheus -v $SCRIPT_DIR/prometheus:/etc/prometheus/ -d --network host prom/prometheus
docker run --rm --name pgw -d --network host prom/pushgateway
docker run --rm --name alertmanager -v $SCRIPT_DIR/alertmanager:/etc/alertmanager -d --network host prom/alertmanager

# Check that prometheus and pgw didn't crash
sleep 10
docker inspect prometheus >/dev/null
docker inspect pgw >/dev/null
docker inspect alertmanager >/dev/null

docker ps
