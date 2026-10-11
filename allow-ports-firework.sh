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
   # Java 閻庣敻鍋婇崰妤€锕㈤崶顒€绀夐柨娑樺娴煎倿鏌?(server-services-registry.sh)
   10081  # gateway
   10082  # mall-catalog-service
   10083  # mall-user-service
   10084  # mall-trade-service
   10085  # platform-service
   10095  # mall-social-service
   10101  # mes-service
   10102  # digital-twin-service
   10103  # forecasting-service
   10105  # scm-inventory-service
   10107  # scm-logistics-service
   10110  # finance-service
   10114  # wallet-service
   10116  # production-service
   10117  # route-optimizer
   10118  # traceability-service
    10122  # data-pipeline
    10125  # erp-management-service
)

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw 闂佸搫鐗滄禍婊堟偩閵娧勫晳闁告侗鍠栭悘妤€鈽夐幘宕囆㈢憸鐗堢叀閹粙鈥﹂幒鏃傤槷闁荤姴娲ら崲鏌ュ储濞戞埃鍋撻悷閭︽Ъ妞?闂佸憡鍑归崹鎶藉极?ufw闂佹寧绋戦張顒勫垂閵娾晛缁╅柛锔诲幗閺嗗繐霉閿濆洤啸婵炲牊鍨垮濂告嚑椤戣法纾绘繝鈷€鍕噧濞村吋鍔欏畷妤呭礄閵堝棗顏? >&2
  exit 1
fi

for p in "${PORTS[@]}"; do
  ufw allow "${p}/tcp" >/dev/null
  echo "allowed ${p}/tcp"
done

ufw status verbose || true
