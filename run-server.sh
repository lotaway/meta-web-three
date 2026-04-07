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

echo "==> Building backend modules with tests skipped"
(cd "$SERVER_DIR" && mvn clean install -DskipTests)

echo "==> Starting services"
for module in "${modules[@]}"; do
  log_file="$LOG_DIR/${module}.log"
  echo "Starting $module (log: $log_file)"
  (
    cd "$SERVER_DIR/$module"
    exec mvn spring-boot:run
  ) >"$log_file" 2>&1 &
done

echo "==> All start commands submitted"
echo "Logs are under $LOG_DIR"
