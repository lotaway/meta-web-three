#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT_DIR/server"
LOG_DIR="$ROOT_DIR/logs/run-server"

mkdir -p "$LOG_DIR"

modules=(
  gateway
  product-service
  user-service
  message-service
  order-service
  payment-service
  commission-service
  media-service
  cart-service
  promotion-service
)

pids=()

echo "==> Stopping any existing services on ports 10081-10092"
for port in 10081 10082 10083 10084 10085 10086 10087 10088 10089 10090 10091 10092; do
  pid=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "  Killing PID $pid on port $port"
    kill $pid 2>/dev/null || true
  fi
done
sleep 2

echo "==> Building backend modules with tests skipped"
(cd "$SERVER_DIR" && mvn clean install -DskipTests)

echo "==> Starting services"
for module in "${modules[@]}"; do
  log_file="$LOG_DIR/${module}.log"
  (
    cd "$SERVER_DIR/$module"
    exec mvn spring-boot:run
  ) >"$log_file" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "Starting $module (PID: $pid, log: $log_file)"
done

echo "==> 所有服务启动命令已提交 (PID 记录完成，Ctrl+C 停止)"
echo "日志目录: $LOG_DIR"
echo "等待 Spring Boot 服务启动 (约 1-3 分钟，可 tail -f *.log 观察)..."
sleep 60  # 初始等待

# 检查启动日志 (Started Application)
echo "检查服务启动状态..."
for module in "${modules[@]}"; do
  log_file="$LOG_DIR/${module}.log"
  if timeout 300 tail -f "$log_file" 2>/dev/null | grep -q "Started ${module//-}Application in"; then
    echo "✓ $module 启动成功"
  else
    echo "⚠ $module 启动中或失败 (检查 $log_file)"
  fi
done 2>/dev/null || true

echo "==> 开始监控服务存活状态 (每10s 检查)..."

trap 'echo "==> 关闭服务中..."; for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
  fi
done
echo "==> 所有服务已停止"
exit 0' INT TERM EXIT

while true; do
  alive=true
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "==> 服务 $pid 已停止，脚本退出"
      exit 1
    fi
  done
  sleep 10
  printf "当前状态 [%s]: %d 服务存活\n" "$(date '+%H:%M:%S')" "${#pids[@]}"
done
