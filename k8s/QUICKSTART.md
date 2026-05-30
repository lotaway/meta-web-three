# Meta Web Three Kubernetes 快速开始指南

## 🚀 5分钟快速部署

### 前置条件

1. **Kubernetes 集群**
   ```bash
   # 检查集群状态
   kubectl cluster-info
   ```

2. **kubectl 工具**
   ```bash
   # 检查 kubectl 版本
   kubectl version --client
   ```

3. **Docker**
   ```bash
   # 检查 Docker 状态
   docker info
   ```

### 快速部署步骤

#### 1. 克隆项目
```bash
git clone <your-repo-url>
cd meta-web-three
```

#### 2. 构建镜像
```bash
# 使用部署脚本构建所有镜像
./k8s/deploy.sh build
```

#### 3. 创建 Secret（重要！）
```bash
# 创建数据库 Secret
./k8s/deploy.sh secret
```

#### 4. 一键部署
```bash
# 部署所有服务
./k8s/deploy.sh install
```

#### 5. 检查状态
```bash
# 查看部署状态
./k8s/deploy.sh status
```

#### 6. 访问应用
```bash
# 端口转发到前端
kubectl port-forward service/client-service 30001:30001 -n meta-web-three

# 端口转发到API
kubectl port-forward service/product-service 10082:10082 -n meta-web-three
```

访问地址：
- 前端: http://localhost:30001
- API: http://localhost:10082

## 📋 详细部署步骤

### 步骤 1: 准备环境

#### 创建存储目录
```bash
# 在 Kubernetes 节点上创建存储目录
sudo mkdir -p /data/{mysql,redis,server/{product,user,order,message}}
sudo chmod 755 /data -R
```

#### 构建应用镜像
```bash
# 构建前端镜像
cd apps/digital-twin/system-management
docker build -t meta-web-three/digital-twin-frontend:latest -f docker/Dockerfile .

# 构建后端服务镜像（上下文为仓库根目录，见 scripts/server-services-registry.sh）
cd ..
docker build -f server/Dockerfile --target product-service -t meta-web-three/product-service:latest .
# 一次性构建全部 Java 服务:
./k8s/deploy.sh build
# K8s 扩展领域服务（cs、mes、supply-chain、erp 等）:
# kubectl apply -f k8s/services/extended-domain-services.yaml
```

### 步骤 2: 创建 Secret

**重要：不要将敏感信息放在代码仓库中！**

```bash
# 方法1: 使用部署脚本交互式创建
./k8s/deploy.sh secret

# 方法2: 手动创建 Secret
kubectl create secret generic database-secret \
  --from-literal=mysql-root-password=your-password \
  --from-literal=mysql-username=your-username \
  --from-literal=mysql-database=your-database \
  -n meta-web-three

# 方法3: 使用外部 Secret 管理系统
# 例如：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault
```

### 步骤 3: 部署基础设施

```bash
# 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 创建存储类
kubectl apply -f k8s/storage/

# 创建配置
kubectl apply -f k8s/configmaps/

# 部署基础设施服务
kubectl apply -f k8s/infrastructure/
```

### 步骤 4: 部署业务服务

```bash
# 部署后端服务
kubectl apply -f k8s/services/

# 部署前端
kubectl apply -f k8s/frontend/

# 部署 API 网关
kubectl apply -f k8s/api-gateway/
```

### 步骤 5: 验证部署

```bash
# 检查所有资源状态
kubectl get all -n meta-web-three

# 检查 Pod 状态
kubectl get pods -n meta-web-three

# 检查服务状态
kubectl get services -n meta-web-three

# 检查存储状态
kubectl get pv,pvc -n meta-web-three
```

## 🔧 常用操作

### 查看日志
```bash
# 查看特定服务日志
./k8s/deploy.sh logs product
./k8s/deploy.sh logs mysql
./k8s/deploy.sh logs all

# 或者使用 kubectl
kubectl logs -f deployment/product-service -n meta-web-three
```

### 进入容器调试
```bash
# 进入 Pod 调试
kubectl exec -it <pod-name> -n meta-web-three -- /bin/bash

# 例如进入 MySQL Pod
kubectl exec -it $(kubectl get pod -l app=mysql -n meta-web-three -o jsonpath='{.items[0].metadata.name}') -n meta-web-three -- mysql -u root -p
```

### 端口转发
```bash
# 前端服务
kubectl port-forward service/client-service 30001:30001 -n meta-web-three

# 后端服务
kubectl port-forward service/product-service 10082:10082 -n meta-web-three
kubectl port-forward service/user-service 10083:10083 -n meta-web-three
kubectl port-forward service/order-service 10084:10084 -n meta-web-three
kubectl port-forward service/message-service 10085:10085 -n meta-web-three

# 数据库服务
kubectl port-forward service/mysql-service 3306:3306 -n meta-web-three
kubectl port-forward service/redis-service 6379:6379 -n meta-web-three
kubectl port-forward service/rabbitmq-service 15672:15672 -n meta-web-three
```

### 扩缩容
```bash
# 扩展服务副本数
kubectl scale deployment product-service --replicas=3 -n meta-web-three

# 查看 HPA 状态（如果配置了）
kubectl get hpa -n meta-web-three
```

## 🐛 故障排除

### 常见问题

#### 1. Secret 不存在错误
```bash
# 检查 Secret 是否存在
kubectl get secret database-secret -n meta-web-three

# 如果不存在，重新创建
./k8s/deploy.sh secret
```

#### 2. Pod 启动失败
```bash
# 查看 Pod 详情
kubectl describe pod <pod-name> -n meta-web-three

# 查看 Pod 日志
kubectl logs <pod-name> -n meta-web-three
```

#### 3. 服务无法访问
```bash
# 检查服务端点
kubectl get endpoints -n meta-web-three

# 检查服务详情
kubectl describe service <service-name> -n meta-web-three
```

#### 4. 存储问题
```bash
# 检查 PVC 状态
kubectl describe pvc <pvc-name> -n meta-web-three

# 检查 PV 状态
kubectl describe pv <pv-name>
```

#### 5. 镜像拉取失败
```bash
# 检查镜像是否存在
docker images | grep meta-web-three

# 重新构建镜像
./k8s/deploy.sh build
```

### 调试命令

```bash
# 查看集群事件
kubectl get events -n meta-web-three --sort-by='.lastTimestamp'

# 查看资源使用情况
kubectl top pods -n meta-web-three
kubectl top nodes

# 查看配置
kubectl get configmap app-config -n meta-web-three -o yaml
kubectl get secret database-secret -n meta-web-three -o yaml
```

## 📊 监控和日志

### 查看应用状态
```bash
# 查看所有资源状态
./k8s/deploy.sh status

# 查看实时日志
./k8s/deploy.sh logs all
```

### 访问管理界面
```bash
# RabbitMQ 管理界面
kubectl port-forward service/rabbitmq-service 15672:15672 -n meta-web-three
# 访问: http://localhost:15672 (admin/admin123)
```

## 🔄 更新和回滚

### 更新应用
```bash
# 构建新镜像
./k8s/deploy.sh build

# 更新 Deployment
kubectl set image deployment/product-service product-service=meta-web-three/product-service:latest -n meta-web-three
```

### 回滚应用
```bash
# 查看部署历史
kubectl rollout history deployment/product-service -n meta-web-three

# 回滚到上一个版本
kubectl rollout undo deployment/product-service -n meta-web-three

# 回滚到指定版本
kubectl rollout undo deployment/product-service --to-revision=2 -n meta-web-three
```

## 🧹 清理资源

### 卸载应用
```bash
# 一键卸载
./k8s/deploy.sh uninstall

# 或者手动删除
kubectl delete -f k8s/deploy-all.yaml
```

### 清理存储
```bash
# 删除 PVC（会删除数据）
kubectl delete pvc --all -n meta-web-three

# 删除 PV
kubectl delete pv --all

# 删除存储目录
sudo rm -rf /data/{mysql,redis,server}
```

## 📚 更多资源

- [完整部署文档](README.md)
- [配置映射详解](MAPPING.md)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [Helm 文档](https://helm.sh/docs/)

## 🆘 获取帮助

如果遇到问题：

1. 查看 [故障排除](#故障排除) 部分
2. 检查 [完整部署文档](README.md)
3. 查看 [配置映射详解](MAPPING.md)
4. 提交 Issue 到项目仓库

---

**快速开始完成！** 🎉

现在你可以访问应用并开始使用了。如果遇到任何问题，请参考故障排除部分或查看详细文档。 