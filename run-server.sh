#!/usr/bin/env bash

set -euo pipefail

# Local default is dev; pass --prod explicitly for production.
if [[ "$*" == *--prod* ]]; then
  export SPRING_PROFILES_ACTIVE=prod
else
  # Local startup must use dev; production has no local datasource URL.
  export SPRING_PROFILES_ACTIVE=dev
fi

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) NULL_DEV="NUL" ;;
  *) NULL_DEV="/dev/null" ;;
esac

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
  pid=$(lsof -ti :"$port" 2>"$NULL_DEV" || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill $pid 2>"$NULL_DEV" || true
  fi
done < <(server_all_ports | sort -u)
sleep 2

echo "==> Installing event-sdk"
(cd "$ROOT_DIR/shared/event-sdk" && mvn install -Dmaven.test.skip=true -q)

echo "==> Compiling then Install common"
(cd "$SERVER_DIR" && mvn clean install -pl common -Dmaven.test.skip=true -q)

echo "==> Building backend modules with tests skipped"
(cd "$SERVER_DIR" && mvn install -Dmaven.test.skip=true -T 4)

echo "==> Starting services"
for entry in "${SERVER_JAVA_SERVICES[@]}"; do
  name="$(server_service_name "$entry")"
  module="$(server_service_module "$entry")"
  log_file="$LOG_DIR/${name}.log"
  jar_file=$(find "$SERVER_DIR/$module/target" -maxdepth 1 -name "$name-*.jar" 2>"$NULL_DEV" | head -1 || true)
  if [ -z "$jar_file" ]; then
    echo "⚠ $name: JAR not found at $module/target/ (skip)"
    continue
  fi
  java -Dspring.profiles.active="$SPRING_PROFILES_ACTIVE" -DAPPLICATION_NAME="$name" -XX:TieredStopAtLevel=1 -jar "$jar_file" >"$log_file" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "Starting $name (PID: $pid)"
done

echo "==> 所有服务启动命令已提交 (PID 记录完成，Ctrl+C 停止)"
echo "日志目录: $LOG_DIR"
echo "等待 Spring Boot 服务启动 (最多约 90 秒，可 tail -f *.log 观察)..."
sleep 5

echo "检查服务启动状态..."

for entry in "${SERVER_JAVA_SERVICES[@]}"; do
  name="$(server_service_name "$entry")"
  log_file="$LOG_DIR/${name}.log"
  started=false
  for _ in {1..17}; do
    if grep -qE "APPLICATION FAILED TO START|Error: Unable to access jarfile|Could not find or load main class" "$log_file" 2>"$NULL_DEV"; then
      echo "✗ $name 启动失败 (检查 $log_file)"
      started=true
      break
    fi
    if grep -q "Started .*Application in" "$log_file" 2>"$NULL_DEV"; then
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
  if kill -0 "$pid" 2>"$NULL_DEV"; then
    kill -TERM "$pid" 2>"$NULL_DEV"
    wait "$pid" 2>"$NULL_DEV" || true
  fi
done
echo "==> 所有服务已停止"
exit 0' INT TERM EXIT

while true; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>"$NULL_DEV"; then
      alive=$((alive + 1))
    else
      echo "  ⚠ 服务进程 PID $pid 已退出 (其余服务继续运行)"
    fi
  done
  if [ "$alive" -eq 0 ]; then
    echo "==> 所有服务均已退出，脚本结束"
    exit 1
  fi
  sleep 10
  printf "当前状态 [%s]: %d/%d 服务存活 (Ctrl+C 停止全部)\n" "$(date '+%H:%M:%S')" "$alive" "${#pids[@]}"
done
