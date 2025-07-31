# Meta Web Three Kubernetes 部署指南

## 概述

本文档描述了如何将 Meta Web Three 项目从 Docker Compose 迁移到 Kubernetes 集群。

## 目录结构

```
k8s/
├── namespace.yaml                    # 命名空间定义
├── configmaps/
│   └── app-config.yaml              # 应用配置
├── secrets/
│   └── database-secret.yaml         # 数据库密钥（模板）
├── storage/
│   └── persistent-volumes.yaml      # 持久化存储卷
├── infrastructure/                  # 基础设施服务
│   ├── zookeeper.yaml
│   ├── mysql.yaml
│   ├── redis.yaml
│   └── rabbitmq.yaml
├── services/                        # 业务服务
│   ├── product-service.yaml
│   ├── user-service.yaml
│   ├── order-service.yaml
│   └── message-service.yaml
├── frontend/
│   └── client.yaml                  # 前端应用
├── api-gateway/
│   └── ingress.yaml                 # API网关
├── deploy-all.yaml                  # 一键部署配置
└── README.md                        # 本文档
```

## 重要概念说明

### 1. Kubernetes 命名规则

- **metadata.name**: 资源的唯一标识符，用于在集群中识别该资源
- **metadata.labels.app**: 标签，用于组织和选择资源，Service 通过 selector 匹配这些标签
- **spec.containers.name**: 容器名称，用于在 Pod 内识别容器

### 2. Volumes 类型说明

在 Kubernetes 中，volumes 分为两种类型：

#### A. 临时存储 (emptyDir)
```yaml
volumes:
- name: temp-data
  emptyDir: {}  # Pod 删除时数据丢失
```

#### B. 持久化存储 (PersistentVolume)
```yaml
volumes:
- name: persistent-data
  persistentVolumeClaim:
    claimName: my-pvc  # 引用 PVC
```

**为什么需要两种类型？**
- **临时存储**: 用于 Pod 内部临时数据，Pod 重启时数据丢失
- **持久化存储**: 用于需要长期保存的数据（如数据库、文件上传等）

### 3. 负载均衡机制

#### Service 层负载均衡
```yaml
apiVersion: v1
kind: Service
metadata:
  name: product-service
spec:
  selector:
    app: product-service  # 选择所有带有此标签的 Pod
  ports:
  - port: 10082
    targetPort: 10082
  type: ClusterIP
```

**工作原理**:
1. Service 通过 selector 选择所有匹配的 Pod
2. 自动为每个 Pod 创建 Endpoint
3. 请求到达 Service 时，自动分发到后端 Pod
4. 默认使用轮询算法进行负载均衡

#### Ingress 层路由分发
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
spec:
  rules:
  - host: api.meta-web-three.local
    http:
      paths:
      - path: /product
        pathType: Prefix
        backend:
          service:
            name: product-service
            port:
              number: 10082
```

**工作原理**:
1. Ingress 根据路径规则将请求路由到不同的 Service
2. Service 再将请求负载均衡到后端 Pod
3. 形成两层负载均衡：Ingress → Service → Pod

## 配置映射说明

### Docker Compose 到 Kubernetes 的映射

| Docker Compose 配置 | Kubernetes 配置 | 说明 |
|-------------------|----------------|------|
| `networks` | `Namespace` + `Service` | 使用命名空间隔离，Service 提供网络发现 |
| `ports` | `Service` + `Ingress` | Service 提供集群内访问，Ingress 提供外部访问 |
| `volumes` | `PersistentVolume` + `PersistentVolumeClaim` | 使用 K8s 持久化存储 |
| `environment` | `ConfigMap` + `Secret` | 配置和敏感信息分离管理 |
| `restart: unless-stopped` | `Deployment` | Deployment 自动重启策略 |
| `depends_on` | `initContainers` | 使用初始化容器等待依赖服务就绪 |

### 无法直接映射的配置

1. **Docker Compose 的 `build` 指令**
   - **问题**: Kubernetes 不直接支持构建镜像
   - **解决方案**: 需要预先构建镜像并推送到镜像仓库
   - **建议**: 使用 CI/CD 流水线自动构建和推送镜像

2. **Docker Compose 的 `container_name`**
   - **问题**: Kubernetes 自动生成 Pod 名称
   - **解决方案**: 使用标签和选择器进行服务发现

3. **Docker Compose 的 `depends_on`**
   - **问题**: Kubernetes 没有直接的依赖关系
   - **解决方案**: 
     - 使用 `initContainers` 等待依赖服务就绪
     - 使用 Helm 的依赖管理
     - 在应用代码中实现重试机制

## 部署步骤

### 1. 前置条件

- Kubernetes 集群（1.20+）
- kubectl 命令行工具
- 镜像仓库访问权限
- 预构建的应用镜像

### 2. 创建 Secret（重要！）

**不要将敏感信息放在代码仓库中！**

```bash
# 方法1: 使用 kubectl 创建 Secret
kubectl create secret generic database-secret \
  --from-literal=mysql-root-password=your-password \
  --from-literal=mysql-username=your-username \
  --from-literal=mysql-database=your-database \
  -n meta-web-three

# 方法2: 使用外部 Secret 管理系统
# 例如：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault
```

### 3. 构建和推送镜像

```bash
# 构建前端镜像
cd client
docker build -t meta-web-three/client:latest .
docker push meta-web-three/client:latest

# 构建后端服务镜像
cd ../server/product-service
docker build -t meta-web-three/product-service:latest .
docker push meta-web-three/product-service:latest

# 重复其他服务...
```

### 4. 创建存储目录

```bash
# 在 Kubernetes 节点上创建存储目录
sudo mkdir -p /data/{mysql,redis,server/{product,user,order,message}}
sudo chmod 755 /data -R
```

### 5. 部署应用

```bash
# 一键部署所有服务
kubectl apply -f k8s/deploy-all.yaml

# 或者分步部署
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storage/
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/infrastructure/
kubectl apply -f k8s/services/
kubectl apply -f k8s/frontend/
kubectl apply -f k8s/api-gateway/
```

### 6. 验证部署

```bash
# 检查命名空间
kubectl get namespace meta-web-three

# 检查所有资源
kubectl get all -n meta-web-three

# 检查持久化存储
kubectl get pv,pvc -n meta-web-three

# 检查服务状态
kubectl get services -n meta-web-three

# 检查 Pod 状态
kubectl get pods -n meta-web-three
```

## 配置说明

### 环境变量和配置

- **ConfigMap**: 存储非敏感配置（如应用配置、数据库连接字符串等）
- **Secret**: 存储敏感信息（如数据库密码、API密钥等）
- **环境变量**: 通过 `env` 或 `envFrom` 注入到容器中

### 存储配置

- **StorageClass**: 定义存储类型（本地存储）
- **PersistentVolume**: 预分配存储空间
- **PersistentVolumeClaim**: 应用申请存储空间

### 网络配置

- **Service**: 提供集群内服务发现和负载均衡
- **Ingress**: 提供外部访问入口
- **NetworkPolicy**: 控制 Pod 间网络通信（可选）

## 监控和日志

### 健康检查

所有服务都配置了：
- **Liveness Probe**: 检测应用是否存活
- **Readiness Probe**: 检测应用是否就绪

### 资源限制

每个服务都配置了资源请求和限制：
- **CPU**: 100m-500m
- **内存**: 128Mi-1Gi

## 扩展和优化建议

### 1. 使用 Helm Chart

建议将配置转换为 Helm Chart，便于版本管理和环境隔离：

```bash
# 创建 Helm Chart
helm create meta-web-three
# 修改 values.yaml 和模板文件
```

### 2. 添加监控

集成 Prometheus + Grafana 监控：
- 应用指标监控
- 资源使用监控
- 业务指标监控

### 3. 配置 HPA

添加水平 Pod 自动扩缩容：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: product-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: product-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 4. 使用 StatefulSet

对于有状态服务（如数据库），考虑使用 StatefulSet：

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql-service
  replicas: 1
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        # ... 其他配置
  volumeClaimTemplates:
  - metadata:
      name: mysql-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: local-storage
      resources:
        requests:
          storage: 10Gi
```

## 故障排除

### 常见问题

1. **Pod 启动失败**
   ```bash
   kubectl describe pod <pod-name> -n meta-web-three
   kubectl logs <pod-name> -n meta-web-three
   ```

2. **服务无法访问**
   ```bash
   kubectl get endpoints -n meta-web-three
   kubectl describe service <service-name> -n meta-web-three
   ```

3. **存储问题**
   ```bash
   kubectl describe pvc <pvc-name> -n meta-web-three
   kubectl describe pv <pv-name>
   ```

### 调试命令

```bash
# 进入 Pod 调试
kubectl exec -it <pod-name> -n meta-web-three -- /bin/bash

# 端口转发
kubectl port-forward service/mysql-service 3306:3306 -n meta-web-three

# 查看事件
kubectl get events -n meta-web-three --sort-by='.lastTimestamp'
```

## 安全建议

1. **使用 RBAC**: 配置适当的角色和权限
2. **网络策略**: 限制 Pod 间通信
3. **镜像安全**: 使用安全的基础镜像，定期更新
4. **密钥管理**: 使用外部密钥管理系统（如 Vault）
5. **TLS 证书**: 为 Ingress 配置 HTTPS

## 备份和恢复

### 数据库备份

```bash
# 创建备份 Job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: mysql-backup
  namespace: meta-web-three
spec:
  template:
    spec:
      containers:
      - name: backup
        image: mysql:8.0
        command:
        - mysqldump
        - -h
        - mysql-service
        - -u
        - root
        - -p123123
        - metawebthree
        - > /backup/backup.sql
        volumeMounts:
        - name: backup-volume
          mountPath: /backup
      volumes:
      - name: backup-volume
        persistentVolumeClaim:
          claimName: backup-pvc
      restartPolicy: Never
EOF
```

## 总结

这个 Kubernetes 配置完整映射了原有的 Docker Compose 配置，并添加了 Kubernetes 的最佳实践：

- ✅ 完整的服务发现和负载均衡
- ✅ 持久化存储配置
- ✅ 配置和密钥管理
- ✅ 健康检查和资源限制
- ✅ 外部访问入口
- ✅ 命名空间隔离
- ✅ 依赖服务等待机制

需要手动处理的部分：
- 🔧 镜像构建和推送
- 🔧 存储目录创建
- 🔧 域名和证书配置
- 🔧 监控和日志收集
- 🔧 备份策略
- 🔧 **Secret 管理（重要！）** 