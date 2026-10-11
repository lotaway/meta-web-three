#!/usr/bin/env bash
set -euo pipefail

# Allow all default external ports (TCP) exposed by docker-compose.env.yml.
# Before running, ensure the server has UFW enabled.


PORTS=(
  2181   # zookeeper
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
   # Java 鐎甸偊鍠楀﹢鍥礉閿涘嫷浼傞柨?(server-services-registry.sh)
   10081  # gateway
   10082  # mall-catalog-service
   10083  # mall-user-service
   10084  # mall-trade-service
   10085  # message-service
   10087  # media-service
   10088  # commission-service
   10092  # cs-service
   10095  # mall-social-service
   10099  # data-analysis-service
   10101  # mes-service
   10102  # digital-twin-service
   10103  # forecasting-service
   10105  # scm-inventory-service
   10107  # scm-logistics-service
   10110  # finance-service
   10114  # wallet-service
   10115  # developer-portal-service
   10116  # production-service
   10117  # route-optimizer
   10118  # traceability-service
    10122  # data-pipeline
    10125  # erp-management-service
    10126  # tenant-service
)

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw 闁哄牜浜滈悾銊ф啑閸涱喖鐏楀☉鎾崇Т瑜版煡鎮介…鎺旂閻犲洤鍢查崢娑氣偓鐟邦槼椤?闁告凹鍨抽弫?ufw闁挎稑鏈崹銊╁绩閸︻厽鏆忓ù锝囧Х濞堟垿姊奸懠顑跨磻濠⒀勭懃娴兼劙宕楅崙銈嗗? >&2
  exit 1
fi

for p in "${PORTS[@]}"; do
  ufw allow "${p}/tcp" >/dev/null
  echo "allowed ${p}/tcp"
done

ufw status verbose || true
