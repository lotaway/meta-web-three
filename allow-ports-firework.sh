#!/usr/bin/env bash
set -euo pipefail

# Allow all default external ports (TCP) exposed by docker-compose.env.yml.
# Before running, ensure the server has UFW enabled.


PORTS=(
  2181   # zookeeper
  3306   # mysql
  5432   # postgres
  5001   # ipfs
  27017  # mongodb
  6379   # redis
  9876   # rocketmq namesrv
  10909  # rocketmq broker 1
  10911  # rocketmq broker 2
  8080   # rocketmq console
  9092   # kafka
  9093   # kafka
  80     # nginx
  9200   # elasticsearch http
  9300   # elasticsearch transport
  5601   # kibana
)

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw 未安装或不可用，请先安装/启用 ufw，或改用你的防火墙工具。" >&2
  exit 1
fi

for p in "${PORTS[@]}"; do
  ufw allow "${p}/tcp" >/dev/null
  echo "allowed ${p}/tcp"
done

ufw status verbose || true
