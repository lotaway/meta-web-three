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
   # Java 微服务端口 (server-services-registry.sh)
   10081  # gateway
   10082  # product-service
   10083  # user-service
   10084  # order-service
   10085  # message-service
   10086  # payment-service
   10087  # media-service
   10088  # commission-service
   10089  # cart-service
   10090  # promotion-service
   10091  # user-action-service
   10092  # cs-service
   10093  # after-sale-service
   10094  # group-buying-service
   10095  # live-service
   10096  # mall-supplier-service
   10097  # review-service
   10098  # risk-control-service
   10099  # data-analysis-service
   10100  # social-commerce-service
   10101  # mes-service
   10102  # digital-twin-service
   10103  # forecasting-service
   10104  # recommendation-service
   10105  # inventory-service
   10106  # warehouse-service
   10107  # logistics-service
   10108  # procurement-service
   10109  # supplier-service
   10110  # finance-service
   10111  # invoice-service
   10112  # reporting-service
   10113  # settlement-service
   10114  # wallet-service
   10115  # developer-portal-service
   10116  # production-service
   10117  # route-optimizer
   10118  # traceability-service
   10119  # hrm-service
   10120  # project-service
   10121  # inventory-alert-service
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
