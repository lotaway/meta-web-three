#!/usr/bin/env bash
# 閸氬海顏?Java 瀵邦喗婀囬崝鈩冩暈閸愬矁銆冮敍鍫濆礋娑撯偓閺佺増宓佸┃鎰剁礆
# 閺嶇厧绱? 閺堝秴濮熼崥宄綧aven 濡€虫健鐠侯垰绶?閻╃顕?server/)|HTTP 缁旑垰褰?
# shellcheck disable=SC2034

SERVER_JAVA_SERVICES=(
  "gateway|gateway|10081"
  "mall-catalog-service|mall-domain/mall-catalog-service|10082"
  "mall-user-service|mall-domain/mall-user-service|10083"
  "mall-trade-service|mall-domain/mall-trade-service|10084"
  "platform-service|platform-domain/platform-service|10085"
  "factory-service|factory-domain/factory-service|10101"
  "ai-service|ai-domain/ai-service|10103"
  "scm-inventory-service|supply-chain-domain/scm-inventory-service|10105"
  "scm-logistics-service|supply-chain-domain/scm-logistics-service|10107"
  "finance-service|erp-domain/finance-service|10110"
  "blockchain-service|blockchain-domain/blockchain-service|10114"
  "mall-social-service|mall-domain/mall-social-service|10095"
  "erp-management-service|erp-domain/erp-management-service|10125"
  "data-pipeline|data-pipeline|10122"
)

server_service_name() { echo "${1%%|*}"; }
server_service_module() { echo "${1#*|}" | cut -d'|' -f1; }

# Local port offset applied to every service HTTP port.
#
# The default range (10081-10125) can be reserved by the OS: on Windows the
# Hyper-V/WSL host network service books contiguous 100-port blocks as
# "excluded port ranges", and binding a reserved port fails with
# "Port <n> was already in use" / WSAEACCES even though nothing listens there.
# Set SERVER_PORT_OFFSET to move the whole stack to a free range, e.g.
#   SERVER_PORT_OFFSET=1000 ./run-mall-server.sh --dev   # gateway on 11081
# The offset is 0 by default, so the documented ports stay authoritative.
SERVER_PORT_OFFSET="${SERVER_PORT_OFFSET:-0}"

server_service_port() { echo $(( ${1##*|} + SERVER_PORT_OFFSET )); }

server_all_ports() {
  local e port
  for e in "${SERVER_JAVA_SERVICES[@]}"; do
    port="${e##*|}"
    echo $(( port + SERVER_PORT_OFFSET ))
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

# Repair build outputs that would break the service startup:
#   - half-written boot jars (an interrupted build leaves them without BOOT-INF)
#   - classes carrying the "Unresolved compilation problem" marker, produced by
#     the Eclipse/JDT compiler of the VS Code Java language server. Maven then
#     reports "Nothing to compile - all classes are up to date" and ships the
#     broken class inside the boot jar.
server_module_target_from_class() {
  printf '%s\n' "$1" | sed -E 's#(/target/classes/.*)$#/target#'
}

server_repair_broken_jars() {
  local server_dir="$1" jar_file entries
  command -v jar >/dev/null 2>&1 || return 0
  while IFS= read -r jar_file; do
    entries="$(jar tf "$jar_file" 2>/dev/null || true)"
    if [ -z "$entries" ]; then
      echo "  ⚠ unreadable jar, cleaning target: $jar_file"
      rm -rf "$(dirname "$(dirname "$jar_file")")/target"
    elif [ -f "$jar_file.original" ] && [ "$(printf '%s\n' "$entries" | grep -c '^BOOT-INF/')" -eq 0 ]; then
      echo "  ⚠ incomplete boot jar, cleaning target: $jar_file"
      rm -rf "$(dirname "$(dirname "$jar_file")")/target"
    fi
  done < <(find "$server_dir" -maxdepth 4 -type f -path '*/target/*.jar')
}

server_repair_unresolved_classes() {
  local server_dir="$1" class_file module_target
  while IFS= read -r class_file; do
    grep -qa "Unresolved compilation problem" "$class_file" 2>/dev/null || continue
    module_target="$(server_module_target_from_class "$class_file")"
    echo "  ⚠ class compiled with unresolved errors, cleaning target: $module_target"
    rm -rf "$module_target"
    break
  done < <(find "$server_dir" -type f -path '*/target/classes/*.class' 2>/dev/null)
}

server_repair_corrupted_targets() {
  local server_dir="$1"
  [ -d "$server_dir" ] || return 0
  server_repair_broken_jars "$server_dir"
  server_repair_unresolved_classes "$server_dir"
}
