#!/usr/bin/env bash
# =============================================================================
# 本地微服务内存排查脚本
#
# 汇总 run-server.sh 启动的 Java 微服务、Docker/WSL 容器以及系统整体的内存占用，
# 用于定位「为什么一堆微服务吃掉几十 GB 内存」。
#
# 依赖 JDK 自带工具 jps / jcmd（JDK 8+），无需额外安装。
#
# 用法:
#   bash scripts/check-server-memory.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
REGISTRY="$SCRIPT_DIR/server-services-registry.sh"

if [ -f "$REGISTRY" ]; then
  # shellcheck source=server-services-registry.sh
  source "$REGISTRY"
fi

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PLATFORM=windows ;;
  Darwin*)              PLATFORM=mac ;;
  *)                    PLATFORM=linux ;;
esac

if [ -t 1 ]; then
  RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
  CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

# ---------------------------------------------------------------------------
# 工具定位
# ---------------------------------------------------------------------------
find_jdk_tool() {
  local tool="$1" jh
  if command -v "$tool" >/dev/null 2>&1; then command -v "$tool"; return 0; fi
  if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/$tool" ]; then
    echo "$JAVA_HOME/bin/$tool"; return 0
  fi
  for jh in "/c/Program Files/Java"/* "$HOME"/.sdkman/candidates/java/* /usr/lib/jvm/*; do
    [ -x "$jh/bin/$tool" ] && { echo "$jh/bin/$tool"; return 0; }
  done
  return 1
}

JPS="$(find_jdk_tool jps || true)"
JCMD="$(find_jdk_tool jcmd || true)"

if [ -z "$JPS" ] || [ -z "$JCMD" ]; then
  echo -e "${RED}错误: 未找到 JDK 的 jps / jcmd。请安装 JDK 或设置 JAVA_HOME。${NC}"
  exit 1
fi

mb() { awk -v v="${1:-0}" 'BEGIN { printf "%.0f", v/1024 }'; }
gb() { awk -v v="${1:-0}" 'BEGIN { printf "%.1f", v/1048576 }'; }

# ---------------------------------------------------------------------------
# 1. 系统内存概览
# ---------------------------------------------------------------------------
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD} 本地微服务内存排查${NC}"
echo -e "${BOLD}========================================${NC}"
echo -e "${CYAN}[1] 系统内存${NC}"

if [ "$PLATFORM" = windows ]; then
  read -r total_kb free_kb < <(powershell.exe -NoProfile -Command \
    "\$os=Get-CimInstance Win32_OperatingSystem; \"\$(\$os.TotalVisibleMemorySize) \$(\$os.FreePhysicalMemory)\"" 2>/dev/null | tr -d '\r')
elif [ "$PLATFORM" = linux ]; then
  total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  free_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
else
  total_kb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 ))
  free_kb=0
fi
total_kb=${total_kb:-0}
free_kb=${free_kb:-0}
used_kb=$(( total_kb - free_kb ))
printf "  物理内存: %s GB 总 / %s GB 可用 / %s GB 已用\n" \
  "$(gb "$total_kb")" "$(gb "$free_kb")" "$(gb "$used_kb")"

# ---------------------------------------------------------------------------
# 2. JVM 进程（核心）
# ---------------------------------------------------------------------------
echo
echo -e "${CYAN}[2] JVM 进程 (jps: 主类 / 参数来源)${NC}"

# pid -> rss(KB) 表，避免逐进程调用 powershell（统一以 KB 存储）
declare -A RSS_KB=()
if [ "$PLATFORM" = windows ]; then
  while read -r _pid _rss; do
    [ -n "${_pid:-}" ] && [ "${_rss:-}" != "" ] && RSS_KB["$_pid"]="$(( _rss / 1024 ))"
  done < <(powershell.exe -NoProfile -Command \
    "Get-Process java -ErrorAction SilentlyContinue | ForEach-Object { '{0} {1}' -f \$_.Id, \$_.WorkingSet64 }" 2>/dev/null | tr -d '\r')
else
  while read -r _pid _rss; do
    [ -n "${_pid:-}" ] && RSS_KB["$_pid"]="${_rss:-0}"
  done < <(ps -eo pid=,rss= 2>/dev/null)
fi

declare -A NAME_OF=()
declare -A CAT_OF=()
declare -A HEAPC_OF=()
declare -A HEAPU_OF=()
declare -A MAXH_OF=()
declare -A META_OF=()
declare -A RSS_OF=()
declare -A CAT_COUNT=() CAT_RSS=() CAT_HEAP=() CAT_MAX=() CAT_META=()

while read -r pid main args; do
  [ -z "${pid:-}" ] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue

  name=""
  if [[ "$args" =~ -DAPPLICATION_NAME=([^[:space:]]+) ]]; then
    name="${BASH_REMATCH[1]}"
    cat="Spring 微服务"
  elif [[ "$main$args" == *GradleDaemon* || "$main" == *gradle* ]]; then
    name="gradle-daemon"
    cat="Gradle 守护进程"
  elif [[ "$main$args" == *redhat.java* || "$main$args" == *jdt.ls* || \
          "$main$args" == *language-server* || "$main$args" == *kotlin-compiler* ]]; then
    name="java-language-server"
    cat="IDE/SDK 语言服务"
  else
    name="$(basename "$main")"
    cat="其它 JVM"
  fi

  heap_info="$("$JCMD" "$pid" GC.heap_info 2>/dev/null || true)"
  heap_c=""; heap_u=""; meta=""
  if [[ "$heap_info" =~ total\ ([0-9]+)K,\ used\ ([0-9]+)K ]]; then
    heap_c="${BASH_REMATCH[1]}"; heap_u="${BASH_REMATCH[2]}"
  fi
  if [[ "$heap_info" =~ Metaspace[[:space:]]+used\ [0-9]+K,\ committed\ ([0-9]+)K ]]; then
    meta="${BASH_REMATCH[1]}"
  fi

  vm_flags="$("$JCMD" "$pid" VM.flags 2>/dev/null || true)"
  maxh=""
  if [[ "$vm_flags" =~ MaxHeapSize=([0-9]+) ]]; then
    maxh="${BASH_REMATCH[1]}"
    maxh="$(awk -v v="$maxh" 'BEGIN { printf "%.0f", v/1048576 }')"
  fi

  rss="${RSS_KB[$pid]:-0}"

  NAME_OF[$pid]="$name"
  CAT_OF[$pid]="$cat"
  HEAPC_OF[$pid]="$heap_c"
  HEAPU_OF[$pid]="$heap_u"
  MAXH_OF[$pid]="$maxh"
  META_OF[$pid]="$meta"
  RSS_OF[$pid]="$rss"

  if [ "$heap_c" != "" ]; then
    CAT_COUNT[$cat]=$(( ${CAT_COUNT[$cat]:-0} + 1 ))
    CAT_RSS[$cat]=$(( ${CAT_RSS[$cat]:-0} + rss ))
    CAT_HEAP[$cat]=$(( ${CAT_HEAP[$cat]:-0} + heap_c ))
    CAT_MAX[$cat]=$(( ${CAT_MAX[$cat]:-0} + ${maxh:-0} ))
    CAT_META[$cat]=$(( ${CAT_META[$cat]:-0} + ${meta:-0} ))
  fi
done < <("$JPS" -lv 2>/dev/null)

# 明细：先 Spring 服务，再其它
printf "  %-8s %-26s %8s %8s %8s %9s %9s\n" "PID" "服务" "RSS(MB)" "堆用(MB)" "堆提(MB)" "上限(MB)" "Meta(MB)"
printf "  %s\n" "----------------------------------------------------------------------------------------"
for pid in "${!NAME_OF[@]}"; do
  [ "${CAT_OF[$pid]}" = "Spring 微服务" ] && \
    printf "  %-8s %-26s %8s %8s %8s %9s %9s\n" \
      "$pid" "${NAME_OF[$pid]}" "${RSS_OF[$pid]}" \
      "$(mb "${HEAPU_OF[$pid]:-0}")" "$(mb "${HEAPC_OF[$pid]:-0}")" \
      "${MAXH_OF[$pid]:--}" "$(mb "${META_OF[$pid]:-0}")"
done | sort -k3 -n -r
for pid in "${!NAME_OF[@]}"; do
  [ "${CAT_OF[$pid]}" != "Spring 微服务" ] && \
    printf "  %-8s %-26s %8s %8s %8s %9s %9s\n" \
      "$pid" "${NAME_OF[$pid]}" "${RSS_OF[$pid]}" \
      "$(mb "${HEAPU_OF[$pid]:-0}")" "$(mb "${HEAPC_OF[$pid]:-0}")" \
      "${MAXH_OF[$pid]:--}" "$(mb "${META_OF[$pid]:-0}")"
done

echo
echo -e "${CYAN}[3] 按类别汇总${NC}"
printf "  %-20s %6s %12s %12s %14s\n" "类别" "个数" "RSS(GB)" "堆提交(GB)" "堆上限(GB)"
printf "  %s\n" "------------------------------------------------------------------"
for cat in "${!CAT_COUNT[@]}"; do
  [ -n "${CAT_HEAP[$cat]:-}" ] || continue
  printf "  %-20s %6s %12s %12s %14s\n" \
    "$cat" "${CAT_COUNT[$cat]}" \
    "$(awk -v v="${CAT_RSS[$cat]}" 'BEGIN{printf "%.1f", v/1048576}')" \
    "$(awk -v v="${CAT_HEAP[$cat]}" 'BEGIN{printf "%.1f", v/1048576}')" \
    "$(awk -v v="${CAT_MAX[$cat]}" 'BEGIN{printf "%.1f", v/1024}')"
done

# ---------------------------------------------------------------------------
# 4. Docker 容器
# ---------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo
  echo -e "${CYAN}[4] Docker 容器内存${NC}"
  docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | sed 's/^/  /'
else
  echo
  echo -e "${CYAN}[4] Docker 容器内存${NC}"
  echo "  (docker 不可用或未运行，跳过)"
fi

# ---------------------------------------------------------------------------
# 5. 服务启动情况
# ---------------------------------------------------------------------------
if declare -F server_all_names >/dev/null 2>&1; then
  total_services=$(server_all_names | wc -l | tr -d ' ')
  running=0
  for pid in "${!CAT_OF[@]}"; do
    [ "${CAT_OF[$pid]}" = "Spring 微服务" ] && running=$((running + 1))
  done
  echo
  echo -e "${CYAN}[5] 服务启动情况${NC}"
  printf "  注册服务 %s 个，当前运行 %s 个\n" "$total_services" "$running"
  if [ "$running" -lt "$total_services" ]; then
    missing=""
    for n in $(server_all_names); do
      found=0
      for pid in "${!NAME_OF[@]}"; do
        [ "${NAME_OF[$pid]}" = "$n" ] && { found=1; break; }
      done
      [ "$found" -eq 0 ] && missing="$missing $n"
    done
    echo -e "  ${YELLOW}未运行:$missing${NC}"
  fi
fi

echo
echo -e "${BOLD}提示:${NC} Spring 服务未设置 -Xmx 时会按物理内存 1/4 取默认上限；"
echo "      「堆上限」列若远大于「堆用」说明只是预留，但仍会推高 commit/RSS。"
