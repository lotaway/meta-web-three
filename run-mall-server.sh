#!/usr/bin/env bash

set -euo pipefail

if [[ "$*" == *--dev* ]]; then
  export SPRING_PROFILES_ACTIVE=dev
else
  export SPRING_PROFILES_ACTIVE=prod
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT_DIR/server"
LOG_DIR="$ROOT_DIR/logs/run-server"
REGISTRY="$ROOT_DIR/scripts/server-services-registry.sh"

source "$REGISTRY"

MALL_SERVICES=()
for entry in "${SERVER_JAVA_SERVICES[@]}"; do
  name="$(server_service_name "$entry")"
  module="$(server_service_module "$entry")"
  if [[ "$name" == "gateway" || "$module" == mall-domain/* ]]; then
    MALL_SERVICES+=("$entry")
  fi
done

mkdir -p "$LOG_DIR"

pids=()

echo "==> Stopping existing mall services"
while read -r port; do
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill $pid 2>/dev/null || true
  fi
done < <(for e in "${MALL_SERVICES[@]}"; do echo "${e##*|}"; done | sort -u)
sleep 2

echo "==> Installing event-sdk"
(cd "$ROOT_DIR/shared/event-sdk" && mvn install -Dmaven.test.skip=true -q)

echo "==> Installing common"
(cd "$SERVER_DIR" && mvn clean install -pl common -Dmaven.test.skip=true -q)

BUILD_MODULES=()
for e in "${MALL_SERVICES[@]}"; do
  BUILD_MODULES+=("$(server_service_module "$e")")
done
BUILD_MODULES_LIST=$(IFS=,; echo "${BUILD_MODULES[*]}")

echo "==> Building mall services ($BUILD_MODULES_LIST)"
(cd "$SERVER_DIR" && mvn clean install -pl "$BUILD_MODULES_LIST" -am -Dmaven.test.skip=true)

echo "==> Starting mall services"
for entry in "${MALL_SERVICES[@]}"; do
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

echo "==> 所有 mall 服务启动命令已提交"
echo "日志目录: $LOG_DIR"
echo "等待 Spring Boot 服务启动..."
sleep 60

echo "检查服务启动状态..."
for entry in "${MALL_SERVICES[@]}"; do
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
