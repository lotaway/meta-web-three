#!/usr/bin/env bash
# 后端 Java 微服务注册表（单一数据源）
# 格式: 服务名|Maven 模块路径(相对 server/)|HTTP 端口
# shellcheck disable=SC2034

SERVER_JAVA_SERVICES=(
  "gateway|gateway|10081"
  "mall-catalog-service|mall-domain/mall-catalog-service|10082"
  "mall-user-service|mall-domain/mall-user-service|10083"
  "mall-trade-service|mall-domain/mall-trade-service|10084"
  "platform-service|platform-domain/platform-service|10085"
  "mes-service|factory-domain/mes-service|10101"
  "digital-twin-service|factory-domain/digital-twin-service|10102"
  "forecasting-service|ai-domain/forecasting-service|10103"
  "scm-inventory-service|supply-chain-domain/scm-inventory-service|10105"
  "scm-logistics-service|supply-chain-domain/scm-logistics-service|10107"
  "finance-service|erp-domain/finance-service|10110"
  "wallet-service|blockchain-domain/wallet-service|10114"
  "mall-social-service|mall-domain/mall-social-service|10095"
  "production-service|factory-domain/production-service|10116"
  "route-optimizer|ai-domain/route-optimizer|10117"
  "traceability-service|blockchain-domain/traceability-service|10118"
  "erp-management-service|erp-domain/erp-management-service|10125"
  "data-pipeline|data-pipeline|10122"
)

server_service_name() { echo "${1%%|*}"; }
server_service_module() { echo "${1#*|}" | cut -d'|' -f1; }
server_service_port() { echo "${1##*|}"; }

server_all_ports() {
  local e port
  for e in "${SERVER_JAVA_SERVICES[@]}"; do
    port="${e##*|}"
    echo "$port"
  done
}

server_all_names() {
  local e
  for e in "${SERVER_JAVA_SERVICES[@]}"; do
    server_service_name "$e"
  done
}

server_module_path() {
  local name="$1" e n
  for e in "${SERVER_JAVA_SERVICES[@]}"; do
    n="$(server_service_name "$e")"
    if [ "$n" = "$name" ]; then
      echo "${e#*|}" | cut -d'|' -f1
      return 0
    fi
  done
  return 1
}

server_jar_build_path() {
  local name="$1" module
  module="$(server_module_path "$name")" || return 1
  echo "${module}/target/${name}-*.jar"
}
