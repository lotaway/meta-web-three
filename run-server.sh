#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT_DIR/server"
LOG_DIR="$ROOT_DIR/logs/run-server"
REGISTRY="$ROOT_DIR/scripts/server-services-registry.sh"

# shellcheck source=scripts/server-services-registry.sh
source "$REGISTRY"

mkdir -p "$LOG_DIR"

pids=()

echo "==> Stopping existing services"
while read -r port; do
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill $pid 2>/dev/null || true
  fi
done < <(server_all_ports | sort -u)
sleep 2

echo "==> Installing event-sdk"
(cd "$ROOT_DIR/shared/event-sdk" && mvn install -DskipTests -q)

echo "==> Building backend modules with tests skipped"
(cd "$SERVER_DIR" && mvn clean install -DskipTests)

echo "==> Starting services"
for entry in "${SERVER_JAVA_SERVICES[@]}"; do
  name="$(server_service_name "$entry")"
  module="$(server_service_module "$entry")"
  log_file="$LOG_DIR/${name}.log"
  (
    cd "$SERVER_DIR"
    exec mvn spring-boot:run -pl "$module"
  ) >"$log_file" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "Starting $name ($module, PID: $pid, log: $log_file)"
done

echo "==> 所有服务启动命令已提交 (PID 记录完成，Ctrl+C 停止)"
echo "日志目录: $LOG_DIR"
echo "等待 Spring Boot 服务启动 (约 1-3 分钟，可 tail -f *.log 观察)..."
sleep 60

echo "检查服务启动状态..."

for entry in "${SERVER_JAVA_SERVICES[@]}"; do
  name="$(server_service_name "$entry")"
  log_file="$LOG_DIR/${name}.log"
  started=false
  for _ in {1..60}; do
    if grep -q "Started .*Application in" "$log_file" 2>/dev/null; then
      echo "✓ $name 启动成功"
      started=true
      break
    fi
    sleep 5
  done
  if [ "$started" = false ]; then
    echo "⚠ $name 启动中或失败 (检查 $log_file)"
  fi
done

echo "==> 开始监控服务存活状态 (每10s 检查)..."

trap 'echo "==> 关闭服务中..."; for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
  fi
done
echo "==> 所有服务已停止"
exit 0' INT TERM EXIT

while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "==> 服务 $pid 已停止，脚本退出"
      exit 1
    fi
  done
  sleep 10
  printf "当前状态 [%s]: %d 服务存活\n" "$(date '+%H:%M:%S')" "${#pids[@]}"
done
