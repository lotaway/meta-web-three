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
   # Java 闁诲海鏁婚崑濠囧窗濡も偓閿曘垽宕堕鈧粈澶愭煥濞戞ê顏村ù鐓庡€块弻?(server-services-registry.sh)
   10081  # gateway
   10082  # mall-catalog-service
   10083  # mall-user-service
   10084  # mall-trade-service
   10085  # platform-service
   10095  # mall-social-service
   10101  # mes-service
   10102  # digital-twin-service
   10103  # ai-service
   10105  # scm-inventory-service
   10107  # scm-logistics-service
   10110  # finance-service
   10114  # wallet-service
   10116  # production-service
   10118  # traceability-service
    10122  # data-pipeline
    10125  # erp-management-service
)

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw 闂備礁鎼悧婊勭濠婂牊鍋╅柕濞у嫬鏅抽梺鍛婁緱閸犳牠鎮樺Δ鈧埥澶愬箻瀹曞泦銏㈡喐閻楀牏鍙€闁诡喕绮欓垾锕傚箳閺冨偆妲烽梺鑽ゅТ濞层倝宕查弻銉ュ偍婵炴垶鍩冮崑鎾绘偡闁附歇濡?闂備礁鎲￠崙褰掑垂閹惰棄鏋?ufw闂備焦瀵х粙鎴﹀嫉椤掑嫬鍨傞柕濞炬櫅缂佲晠鏌涢敂璇插箺闁哄棗绻愰湁闁挎繂娲ゅ暩濠电偛鐗婇崹鍨潖婵傚憡鍤戞い鎴ｆ硶绾剧粯绻濋埛鈧崟顓熷櫑婵炴潙鍚嬮崝娆忕暦濡ゅ懎绀勯柕鍫濇椤? >&2
  exit 1
fi

for p in "${PORTS[@]}"; do
  ufw allow "${p}/tcp" >/dev/null
  echo "allowed ${p}/tcp"
done

ufw status verbose || true
