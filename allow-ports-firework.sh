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
   # Java 闂佽娴烽弫濠氬磻婵犲洤绐楁俊銈傚亾闁挎洏鍨藉畷鍫曨敆閳ь剛绮堟径鎰叆婵炴垶锚椤忔潙霉閻撳骸鈧潡寮?(server-services-registry.sh)
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
   10114  # blockchain-service
   10116  # production-service
    10122  # data-pipeline
    10125  # erp-management-service
)

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw 闂傚倷绀侀幖顐︽偋濠婂嫮顩叉繝濠傜墛閸嬧晠鏌曟繛褍瀚弲鎶芥⒑閸涘﹣绶遍柛鐘崇墵閹ê螖閳ь剟鍩ユ径鎰鐎规洖娉﹂姀銏″枑闁绘鐗忛崣鈧梺璇″枙缁瑩鍨鹃敃鍌氱闁哄啫鍋嗗Σ鐑芥⒑閼姐倕孝婵炲眰鍊濆畷鏌ュ蓟閵夈儱鍋嶅┑鐐村灦閸╁啴宕戦幘缁樺仭闂侇叏闄勬瓏婵?闂傚倷绀侀幉锟犲礄瑜版帒鍨傞柟鎯版閺?ufw闂傚倷鐒︾€笛呯矙閹达箑瀚夋い鎺戝閸ㄥ倿鏌曟繛鐐珔缂備讲鏅犻弻娑㈡晜鐠囨彃绠洪梺鍝勬缁绘劙婀侀梺鎸庣箓濞层倕鏆╂繝鐢靛仜閻楀﹪宕归崹顕呮綎濠靛倸鎲￠崵鎴炪亜閹达絾纭剁痪鍓х帛缁绘繈鍩涢埀顒勫礋椤撶喎娅戝┑鐐存綑閸氬宕濆▎蹇曟殾婵°倕鎳庣粈鍕煏閸繃顥滄い? >&2
  exit 1
fi

for p in "${PORTS[@]}"; do
  ufw allow "${p}/tcp" >/dev/null
  echo "allowed ${p}/tcp"
done

ufw status verbose || true
