#!/bin/bash
# =============================================================================
# 领域边界检查脚本
# 检查服务是否访问了非本领域的数据表
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config/domain"
ERRORS=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "领域边界检查 (Domain Boundary Check)"
echo "========================================"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${RED}错误: 配置目录不存在: $CONFIG_DIR${NC}"
    exit 1
fi

# 获取服务名称（从命令行参数或从当前目录推断）
SERVICE_NAME="${1:-}"

if [ -z "$SERVICE_NAME" ]; then
    # 尝试从当前目录推断服务名
    CURRENT_DIR=$(basename "$PWD")
    
    # 检查是否在某个服务目录下
    for config_file in "$CONFIG_DIR"/*.yaml; do
        if [ -f "$config_file" ]; then
            service=$(basename "$config_file" .yaml)
            if [[ "$CURRENT_DIR" == *"$service"* ]] || [[ "$service" == *"$CURRENT_DIR"* ]]; then
                SERVICE_NAME="$service"
                break
            fi
        fi
    done
fi

if [ -z "$SERVICE_NAME" ]; then
    echo -e "${YELLOW}用法: $0 <服务名>${NC}"
    echo "可用服务:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml
    exit 0
fi

echo -e "\n检查服务: ${GREEN}$SERVICE_NAME${NC}"
echo "----------------------------------------"

# 查找该服务所属的领域配置
DOMAIN_CONFIG=""
SERVICE_ALLOWED_TABLES=()

for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config_file" ]; then
        # 检查该配置文件中是否有此服务
        if grep -q "^  $SERVICE_NAME:" "$config_file" 2>/dev/null || \
           grep -q "services:" "$config_file"; then
            DOMAIN_CONFIG="$config_file"
            break
        fi
    fi
done

if [ -z "$DOMAIN_CONFIG" ]; then
    # 尝试直接匹配服务名作为领域名
    if [ -f "$CONFIG_DIR/${SERVICE_NAME}.yaml" ]; then
        DOMAIN_CONFIG="$CONFIG_DIR/${SERVICE_NAME}.yaml"
    fi
fi

if [ -z "$DOMAIN_CONFIG" ]; then
    echo -e "${RED}错误: 未找到服务 '$SERVICE_NAME' 的配置${NC}"
    exit 1
fi

echo "配置文件: $(basename "$DOMAIN_CONFIG")"

# 从配置文件中提取允许的表
ALLOWED_TABLES=$(grep -A 50 "services:" "$DOMAIN_CONFIG" | grep -A 20 "$SERVICE_NAME:" | \
    grep "allowed_tables:" -A 10 | tail -n +2 | head -n 10 | sed 's/^[[:space:]]*- //' | grep -v "^$")

if [ -z "$ALLOWED_TABLES" ]; then
    # 检查是否为只读服务
    if grep -A 5 "$SERVICE_NAME:" "$DOMAIN_CONFIG" | grep -q "readonly: true"; then
        echo -e "服务类型: ${YELLOW}只读服务 (readonly)${NC}"
        ALLOWED_TABLES=""
    else
        echo -e "${YELLOW}警告: 无法确定允许的表${NC}"
    fi
else
    echo "允许访问的表:"
    echo "$ALLOWED_TABLES" | sed 's/^/  - /'
fi

echo ""
echo "检查源代码..."

# 需要检查的目录
CHECK_DIRS=()

# 确定要检查的目录
if [ -d "$PROJECT_ROOT/server/$SERVICE_NAME" ]; then
    CHECK_DIRS+=("$PROJECT_ROOT/server/$SERVICE_NAME")
elif [ -d "$PROJECT_ROOT/server/mall-domain/$SERVICE_NAME" ]; then
    CHECK_DIRS+=("$PROJECT_ROOT/server/mall-domain/$SERVICE_NAME")
elif [ -d "$PROJECT_ROOT/server/supply-chain-domain/$SERVICE_NAME" ]; then
    CHECK_DIRS+=("$PROJECT_ROOT/server/supply-chain-domain/$SERVICE_NAME")
elif [ -d "$PROJECT_ROOT/server/platform-domain/$SERVICE_NAME" ]; then
    CHECK_DIRS+=("$PROJECT_ROOT/server/platform-domain/$SERVICE_NAME")
fi

if [ ${#CHECK_DIRS[@]} -eq 0 ]; then
    echo -e "${YELLOW}警告: 未找到服务源代码目录${NC}"
    exit 0
fi

# 检查 SQL 文件
echo -e "\n${YELLOW}检查 SQL 文件...${NC}"
for sql_file in $(find "${CHECK_DIRS[@]}" -name "*.sql" 2>/dev/null); do
    echo "  检查: $sql_file"
    
    # 检查是否有跨域表访问
    for table in $(grep -ohE "(FROM|INTO|UPDATE|DELETE FROM|JOIN)[[:space:]]+[[:alnum:]_]+" "$sql_file" | \
        awk '{print $2}' | sort -u); do
        if [ -n "$table" ]; then
            if [ -n "$ALLOWED_TABLES" ]; then
                if ! echo "$ALLOWED_TABLES" | grep -qw "$table"; then
                    echo -e "    ${RED}错误: 访问了未授权的表: $table${NC}"
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        fi
    done
done

# 检查 Java 代码中的表名
echo -e "\n${YELLOW}检查 Java 代码...${NC}"
for java_file in $(find "${CHECK_DIRS[@]}" -name "*.java" 2>/dev/null); do
    # 检查是否有直接 SQL 操作
    if grep -qE "@Query|@NamedQuery|createQuery|createNativeQuery" "$java_file" 2>/dev/null; then
        # 提取 SQL 语句中的表名
        tables=$(grep -ohE "(FROM|INTO|UPDATE|DELETE FROM)[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*" "$java_file" | \
            awk '{print $2}' | sort -u)
        
        for table in $tables; do
            if [ -n "$ALLOWED_TABLES" ] && ! echo "$ALLOWED_TABLES" | grep -qw "$table"; then
                # 检查是否为关联表（如 order_items 关联 orders）
                owner_domain=$(grep "owner_domain:" "$DOMAIN_CONFIG" | awk '{print $2}')
                # 简化检查：只要不在允许列表中就报错
                echo -e "    ${RED}警告: 代码中可能访问未授权的表: $table${NC} (在 $java_file)"
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi
done

# 检查 Python 代码
echo -e "\n${YELLOW}检查 Python 代码...${NC}"
for py_file in $(find "${CHECK_DIRS[@]}" -name "*.py" 2>/dev/null); do
    if grep -qE "execute\(|cursor\.execute" "$py_file" 2>/dev/null; then
        tables=$(grep -ohE "execute\([^)]*(FROM|INTO|UPDATE|DELETE FROM)[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*" "$py_file" | \
            grep -ohE "(FROM|INTO|UPDATE|DELETE FROM)[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*" | \
            awk '{print $2}' | sort -u)
        
        for table in $tables; do
            if [ -n "$ALLOWED_TABLES" ] && ! echo "$ALLOWED_TABLES" | grep -qw "$table"; then
                echo -e "    ${RED}警告: 代码中可能访问未授权的表: $table${NC} (在 $py_file)"
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi
done

echo ""
echo "========================================"
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}检查失败: 发现 $ERRORS 个问题${NC}"
    exit 1
else
    echo -e "${GREEN}检查通过: 未发现跨域访问问题${NC}"
    exit 0
fi