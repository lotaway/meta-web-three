#!/bin/bash

# Meta Web Three Kubernetes 部署脚本
# 使用方法: ./deploy.sh [install|uninstall|status|logs]

set -e

NAMESPACE="meta-web-three"
K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 kubectl 是否可用
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装或不在 PATH 中"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        exit 1
    fi
    
    log_success "kubectl 连接正常"
}

# 检查 Secret 是否存在
check_secrets() {
    log_info "检查必要的 Secret..."
    
    if ! kubectl get secret database-secret -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secret 'database-secret' 不存在！"
        log_info "请先创建 Secret，可以使用以下命令："
        echo
        echo "kubectl create secret generic database-secret \\"
        echo "  --from-literal=mysql-root-password=your-password \\"
        echo "  --from-literal=mysql-username=your-username \\"
        echo "  --from-literal=mysql-database=your-database \\"
        echo "  -n $NAMESPACE"
        echo
        log_error "请先创建 Secret 后再运行部署"
        exit 1
    else
        log_success "Secret 'database-secret' 存在"
    fi
}

# 检查必要的镜像是否存在
check_images() {
    log_info "检查必要的镜像..."
    
    local images=(
        "meta-web-three/client:latest"
        "meta-web-three/product-service:latest"
        "meta-web-three/user-service:latest"
        "meta-web-three/order-service:latest"
        "meta-web-three/message-service:latest"
    )
    
    for image in "${images[@]}"; do
        if ! docker image inspect "$image" &> /dev/null; then
            log_warning "镜像 $image 不存在，请先构建镜像"
            log_info "可以使用以下命令构建镜像："
            echo "  docker build -t $image ."
        else
            log_success "镜像 $image 存在"
        fi
    done
}

# 创建存储目录
create_storage_dirs() {
    log_info "创建存储目录..."
    
    local dirs=(
        "/data/mysql"
        "/data/redis"
        "/data/server/product"
        "/data/server/user"
        "/data/server/order"
        "/data/server/message"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_info "创建目录: $dir"
            sudo mkdir -p "$dir"
            sudo chmod 755 "$dir"
        else
            log_success "目录已存在: $dir"
        fi
    done
}

# 安装应用
install() {
    log_info "开始安装 Meta Web Three 到 Kubernetes..."
    
    check_kubectl
    check_secrets
    check_images
    create_storage_dirs
    
    log_info "应用 Kubernetes 配置..."
    kubectl apply -f "$K8S_DIR/deploy-all.yaml"
    
    log_info "等待 Pod 启动..."
    kubectl wait --for=condition=ready pod -l app=zookeeper -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=mysql -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=rabbitmq -n "$NAMESPACE" --timeout=300s
    
    log_info "等待业务服务启动..."
    kubectl wait --for=condition=ready pod -l app=product-service -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=user-service -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=order-service -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=message-service -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=client -n "$NAMESPACE" --timeout=300s
    
    log_success "安装完成！"
    show_status
}

# 卸载应用
uninstall() {
    log_info "开始卸载 Meta Web Three..."
    
    check_kubectl
    
    log_info "删除 Kubernetes 资源..."
    kubectl delete -f "$K8S_DIR/deploy-all.yaml" --ignore-not-found=true
    
    log_success "卸载完成！"
}

# 显示状态
status() {
    log_info "检查 Meta Web Three 状态..."
    
    check_kubectl
    
    echo
    log_info "命名空间状态:"
    kubectl get namespace "$NAMESPACE" 2>/dev/null || log_error "命名空间 $NAMESPACE 不存在"
    
    echo
    log_info "Pod 状态:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || log_error "无法获取 Pod 状态"
    
    echo
    log_info "服务状态:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || log_error "无法获取服务状态"
    
    echo
    log_info "持久化存储状态:"
    kubectl get pv,pvc -n "$NAMESPACE" 2>/dev/null || log_error "无法获取存储状态"
    
    echo
    log_info "Ingress 状态:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || log_error "无法获取 Ingress 状态"
    
    echo
    log_info "访问信息:"
    echo "  前端: http://meta-web-three.local"
    echo "  API: http://api.meta-web-three.local"
    echo "  RabbitMQ 管理界面: http://localhost:15672 (admin/admin123)"
    
    echo
    log_info "端口转发命令:"
    echo "  kubectl port-forward service/client-service 30001:30001 -n $NAMESPACE"
    echo "  kubectl port-forward service/product-service 10082:10082 -n $NAMESPACE"
    echo "  kubectl port-forward service/user-service 10083:10083 -n $NAMESPACE"
    echo "  kubectl port-forward service/order-service 10084:10084 -n $NAMESPACE"
    echo "  kubectl port-forward service/message-service 10085:10085 -n $NAMESPACE"
}

# 显示日志
logs() {
    local service=${1:-"all"}
    
    log_info "显示 $service 服务的日志..."
    
    check_kubectl
    
    case $service in
        "zookeeper")
            kubectl logs -f deployment/zookeeper -n "$NAMESPACE"
            ;;
        "mysql")
            kubectl logs -f deployment/mysql -n "$NAMESPACE"
            ;;
        "redis")
            kubectl logs -f deployment/redis -n "$NAMESPACE"
            ;;
        "rabbitmq")
            kubectl logs -f deployment/rabbitmq -n "$NAMESPACE"
            ;;
        "product")
            kubectl logs -f deployment/product-service -n "$NAMESPACE"
            ;;
        "user")
            kubectl logs -f deployment/user-service -n "$NAMESPACE"
            ;;
        "order")
            kubectl logs -f deployment/order-service -n "$NAMESPACE"
            ;;
        "message")
            kubectl logs -f deployment/message-service -n "$NAMESPACE"
            ;;
        "client")
            kubectl logs -f deployment/client -n "$NAMESPACE"
            ;;
        "all")
            log_info "显示所有服务的日志（按 Ctrl+C 停止）..."
            kubectl logs -f -l app -n "$NAMESPACE"
            ;;
        *)
            log_error "未知服务: $service"
            echo "可用服务: zookeeper, mysql, redis, rabbitmq, product, user, order, message, client, all"
            exit 1
            ;;
    esac
}

# 构建镜像
build_images() {
    log_info "构建应用镜像..."
    
    # 构建前端镜像
    log_info "构建前端镜像..."
    cd "$K8S_DIR/../client"
    docker build -t meta-web-three/client:latest .
    
    # 构建后端服务镜像
    local services=("product-service" "user-service" "order-service" "message-service")
    
    for service in "${services[@]}"; do
        log_info "构建 $service 镜像..."
        cd "$K8S_DIR/../server/$service"
        docker build -t "meta-web-three/$service:latest" .
    done
    
    log_success "所有镜像构建完成！"
}

# 推送镜像
push_images() {
    log_info "推送镜像到仓库..."
    
    local images=(
        "meta-web-three/client:latest"
        "meta-web-three/product-service:latest"
        "meta-web-three/user-service:latest"
        "meta-web-three/order-service:latest"
        "meta-web-three/message-service:latest"
    )
    
    for image in "${images[@]}"; do
        log_info "推送镜像: $image"
        docker push "$image"
    done
    
    log_success "所有镜像推送完成！"
}

# 创建 Secret
create_secret() {
    log_info "创建数据库 Secret..."
    
    check_kubectl
    
    echo "请输入数据库配置信息："
    read -p "MySQL Root 密码: " mysql_password
    read -p "MySQL 用户名 (默认: root): " mysql_username
    mysql_username=${mysql_username:-root}
    read -p "MySQL 数据库名 (默认: metawebthree): " mysql_database
    mysql_database=${mysql_database:-metawebthree}
    
    kubectl create secret generic database-secret \
        --from-literal=mysql-root-password="$mysql_password" \
        --from-literal=mysql-username="$mysql_username" \
        --from-literal=mysql-database="$mysql_database" \
        -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secret 创建完成！"
}

# 显示帮助信息
show_help() {
    echo "Meta Web Three Kubernetes 部署脚本"
    echo
    echo "使用方法: $0 [COMMAND]"
    echo
    echo "命令:"
    echo "  install     安装应用"
    echo "  uninstall   卸载应用"
    echo "  status      显示应用状态"
    echo "  logs [SERVICE] 显示服务日志"
    echo "  build       构建所有镜像"
    echo "  push        推送所有镜像"
    echo "  secret      创建数据库 Secret"
    echo "  help        显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 secret    # 首先创建 Secret"
    echo "  $0 install   # 然后安装应用"
    echo "  $0 status    # 查看状态"
    echo "  $0 logs product"
    echo "  $0 logs all"
    echo
    echo "注意: 请先运行 '$0 secret' 创建数据库 Secret，然后再安装应用"
}

# 主函数
main() {
    case ${1:-"help"} in
        "install")
            install
            ;;
        "uninstall")
            uninstall
            ;;
        "status")
            status
            ;;
        "logs")
            logs "$2"
            ;;
        "build")
            build_images
            ;;
        "push")
            push_images
            ;;
        "secret")
            create_secret
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 