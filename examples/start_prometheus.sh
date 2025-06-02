#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
docker run --rm --name prometheus -v $SCRIPT_DIR/prometheus:/etc/prometheus/ -d --network host prom/prometheus
docker run --rm --name pgw -d --network host prom/pushgateway

# Check that prometheus and pgw didn't crash
sleep 10
docker inspect prometheus >/dev/null
docker inspect pgw >/dev/null

docker ps
