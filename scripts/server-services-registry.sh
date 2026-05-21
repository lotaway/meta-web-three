#!/usr/bin/env bash
# 后端 Java 微服务注册表（单一数据源）
# 格式: 服务名|Maven 模块路径(相对 server/)|HTTP 端口
# shellcheck disable=SC2034

SERVER_JAVA_SERVICES=(
  "gateway|gateway|10081"
  "product-service|mall-domain/product-service|10082"
  "user-service|mall-domain/user-service|10083"
  "order-service|mall-domain/order-service|10084"
  "message-service|platform-domain/message-service|10085"
  "payment-service|mall-domain/payment-service|10086"
  "media-service|platform-domain/media-service|10087"
  "commission-service|platform-domain/commission-service|10088"
  "cart-service|mall-domain/cart-service|10089"
  "promotion-service|mall-domain/promotion-service|10090"
  "user-action-service|platform-domain/user-action-service|10091"
  "cs-service|platform-domain/cs-service|10092"
  "mes-service|factory-domain/mes-service|10101"
  "digital-twin-service|factory-domain/digital-twin-service|10102"
  "forecasting-service|ai-domain/forecasting-service|10103"
  "recommendation-service|ai-domain/recommendation-service|10104"
  "inventory-service|supply-chain-domain/inventory-service|10105"
  "warehouse-service|supply-chain-domain/warehouse-service|10106"
  "logistics-service|supply-chain-domain/logistics-service|10107"
  "procurement-service|supply-chain-domain/procurement-service|10108"
  "supplier-service|supply-chain-domain/supplier-service|10109"
  "finance-service|erp-domain/finance-service|10110"
  "invoice-service|erp-domain/invoice-service|10111"
  "reporting-service|erp-domain/reporting-service|10112"
  "settlement-service|erp-domain/settlement-service|10113"
  "wallet-service|blockchain-domain/wallet-service|10114"
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
