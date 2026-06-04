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
  "recommendation-service|mall-domain/recommendation-service|10104"
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
  "after-sale-service|mall-domain/after-sale-service|10093"
  "group-buying-service|mall-domain/group-buying-service|10094"
  "live-service|mall-domain/live-service|10095"
  "mall-supplier-service|mall-domain/mall-supplier-service|10096"
  "review-service|mall-domain/review-service|10097"
  "risk-control-service|mall-domain/risk-control-service|10098"
  "data-analysis-service|platform-domain/data-analysis-service|10099"
  "social-commerce-service|platform-domain/social-commerce-service|10100"
  "developer-portal-service|platform-domain/developer-portal-service|10115"
  "production-service|factory-domain/production-service|10116"
  "route-optimizer|ai-domain/route-optimizer|10117"
  "traceability-service|blockchain-domain/traceability-service|10118"
  "hrm-service|erp-domain/hrm-service|10119"
  "project-service|erp-domain/project-service|10120"
  "inventory-alert-service|supply-chain-domain/inventory-alert-service|10121"
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
