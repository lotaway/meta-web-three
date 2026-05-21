#!/usr/bin/env bash
# 根据 scripts/server-services-registry.sh 生成 docker-compose.server.yml
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=server-services-registry.sh
source "$ROOT_DIR/scripts/server-services-registry.sh"

OUT="$ROOT_DIR/docker-compose.server.yml"

gen_compose_service() {
  local entry="$1" name port vol
  name="$(server_service_name "$entry")"
  port="$(server_service_port "$entry")"
  vol="${name%-service}"
  cat <<EOF
  ${name}:
    <<: *java-service
    build:
      context: .
      dockerfile: server/Dockerfile
      target: ${name}
    container_name: meta-web-three-${name}
    ports:
      - "${port}:${port}"
    volumes:
      - ${vol}_service_data:/server/${vol}
      - ./.aws:/.aws:ro
EOF
}

{
  cat <<'HDR'
x-java-service: &java-service
  environment:
    - SPRING_PROFILES_ACTIVE=dev
    - SPRING_CLOUD_ZOOKEEPER_CONNECT_STRING=zookeeper:2181
  depends_on:
    zookeeper:
      condition: service_healthy
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
  networks:
    - meta-web-three
    - default
  restart: unless-stopped

services:
  gateway:
    build:
      context: .
      dockerfile: server/Dockerfile
      target: gateway
    container_name: meta-web-three-gateway
    ports:
      - "10081:10081"
    environment:
      - SPRING_PROFILES_ACTIVE=dev
      - SPRING_CLOUD_ZOOKEEPER_CONNECT_STRING=zookeeper:2181
    volumes:
      - gateway_data:/server/gateway
      - ./.aws:/.aws:ro
    depends_on:
      zookeeper:
        condition: service_healthy
    networks:
      - meta-web-three
      - default
    restart: unless-stopped
HDR
  for entry in "${SERVER_JAVA_SERVICES[@]}"; do
    name="$(server_service_name "$entry")"
    [ "$name" = "gateway" ] && continue
    gen_compose_service "$entry"
  done
  echo ''
  echo 'volumes:'
  echo '  gateway_data:'
  for entry in "${SERVER_JAVA_SERVICES[@]}"; do
    name="$(server_service_name "$entry")"
    [ "$name" = "gateway" ] && continue
    vol="${name%-service}"
    echo "  ${vol}_service_data:"
  done
} > "$OUT"

echo "Wrote $OUT"
