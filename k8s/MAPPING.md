# Docker Compose 到 Kubernetes 配置映射详解

## 概述

本文档详细说明了如何将 Meta Web Three 项目的 Docker Compose 配置映射到 Kubernetes 配置。

## 1. 网络配置映射

### Docker Compose 网络配置

```yaml
# docker-compose.yml
networks:
  meta-web-three:
    driver: bridge
```

### Kubernetes 网络配置

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: meta-web-three
  labels:
    name: meta-web-three
```

**映射说明**：

- Docker Compose 的 `networks` 映射到 Kubernetes 的 `Namespace`
- 使用命名空间进行网络隔离，比 Docker 网络更安全
- 服务间通信通过 Service 实现，无需显式网络配置

## 2. 端口映射

### Docker Compose 端口配置

```yaml
# docker-compose.yml
services:
  client:
    ports:
      - "${VITE_PORT}:${VITE_PORT}"

# docker-compose.server.yaml
services:
  gateway:
    ports:
      - "10081:10081"
  product-service:
    ports:
      - "10082:10082"
  user-service:
    ports:
      - "10083:10083"
  order-service:
    ports:
      - "10084:10084"
  promotion-service:
    ports:
      - "10090:10090"
  user-action-service:
    ports:
      - "10091:10091"
```

### Kubernetes 端口配置

```yaml
# user-action-service Service 配置
apiVersion: v1
kind: Service
metadata:
  name: user-action-service
spec:
  selector:
    app: user-action-service
  ports:
  - port: 10091
    targetPort: 10091
  type: ClusterIP

# user-action-service Ingress 配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
spec:
  rules:
  - host: api.meta-web-three.local
    http:
      paths:
      - path: /action
        pathType: Prefix
        backend:
          service:
            name: user-action-service
            port:
              number: 10091
```

```yaml
# Service 配置
apiVersion: v1
kind: Service
metadata:
  name: client-service
spec:
  selector:
    app: client
  ports:
  - port: 30001
    targetPort: 30001
  type: ClusterIP

# Ingress 配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: client-ingress
spec:
  rules:
  - host: meta-web-three.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: client-service
            port:
              number: 30001
```

**映射说明**：

- Docker Compose 的 `ports` 映射到 Kubernetes 的 `Service` + `Ingress`
- Service 提供集群内访问
- Ingress 提供外部访问入口
- 更灵活的负载均衡和路由配置

## 3. 环境变量配置

### Docker Compose 环境配置

```yaml
# docker-compose.dataenv.yml
services:
  mysql:
    env_file:
      - .env
      - ./server/.env
    environment:
      - MYSQL_USER=${MYSQL_USER}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
      - MYSQL_DATABASE=${MYSQL_DATABASE}
      - TZ=${TIMEZONE}
```

### Kubernetes 环境配置

```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  application-common-dev.yml: |
    spring:
      datasource:
        url: jdbc:mysql://mysql-service:3306/metawebthree
        username: root
        password: 123123

# Secret
apiVersion: v1
kind: Secret
metadata:
  name: database-secret
type: Opaque
data:
  mysql-root-password: MTIzMTIz
  mysql-username: cm9vdA==

# Deployment 中的环境变量
spec:
  template:
    spec:
      containers:
      - name: mysql
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: mysql-root-password
        - name: MYSQL_DATABASE
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: mysql-database
```

**映射说明**：

- Docker Compose 的 `env_file` 映射到 Kubernetes 的 `ConfigMap`
- Docker Compose 的 `environment` 映射到 Kubernetes 的 `Secret` + 环境变量
- 敏感信息使用 Secret 存储，非敏感信息使用 ConfigMap
- 更安全的配置管理

## 4. 存储卷配置

### Docker Compose 存储配置

```yaml
# docker-compose.dataenv.yml
services:
  mysql:
    volumes:
      - ${DATA_PATH}/.data/mysql:/var/lib/mysql
  redis:
    volumes:
      - ${DATA_PATH}/.data/redis:/data

# docker-compose.server.yaml
services:
  product-service:
    volumes:
      - ${DATA_PATH}/.data/server/product:/server/product
  user-service:
    volumes:
      - ${DATA_PATH}/.data/server/user:/server/user
  order-service:
    volumes:
      - ${DATA_PATH}/.data/server/order:/server/order
  message-service:
    volumes:
      - ${DATA_PATH}/.data/server/message:/server/message
  user-action-service:
    volumes:
      - ${DATA_PATH}/.data/server/user-action:/server/user-action
```

### Kubernetes 存储配置

```yaml
# StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer

# PersistentVolume
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mysql-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /data/mysql
  storageClassName: local-storage

# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-storage
  resources:
    requests:
      storage: 10Gi

# Deployment 中的卷挂载
spec:
  template:
    spec:
      containers:
      - name: mysql
        volumeMounts:
        - name: mysql-data
          mountPath: /var/lib/mysql
      volumes:
      - name: mysql-data
        persistentVolumeClaim:
          claimName: mysql-pvc
```

**映射说明**：

- Docker Compose 的 `volumes` 映射到 Kubernetes 的 `PersistentVolume` + `PersistentVolumeClaim`
- 使用 StorageClass 定义存储类型
- 更灵活的存储管理和动态分配
- 支持多种存储后端

## 5. 重启策略

### Docker Compose 重启配置

```yaml
# docker-compose.server.yaml
services:
  product-service:
    restart: unless-stopped
  user-service:
    restart: unless-stopped
  order-service:
    restart: unless-stopped
  message-service:
    restart: unless-stopped
  user-action-service:
    restart: unless-stopped
```

### Kubernetes 重启配置

```yaml
# Deployment 自动重启
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: product-service
  template:
    metadata:
      labels:
        app: product-service
    spec:
      containers:
        - name: product-service
          image: meta-web-three/product-service:latest
```

**映射说明**：

- Docker Compose 的 `restart: unless-stopped` 映射到 Kubernetes 的 `Deployment`
- Deployment 自动管理 Pod 的生命周期
- 支持滚动更新和回滚
- 更强大的故障恢复能力

## 6. 依赖关系

### Docker Compose 依赖配置

```yaml
# docker-compose.server.yaml (注释掉的配置)
services:
  product-service:
    # depends_on:
    #   zookeeper:
    #     - condition: service_healthy
    #   mysql:
    #     - condition: service_healthy
    #   redis:
    #     condition: service_started
```

### Kubernetes 依赖配置

```yaml
# 使用 initContainers 等待依赖服务
spec:
  template:
    spec:
      initContainers:
        - name: wait-for-mysql
          image: busybox
          command:
            [
              "sh",
              "-c",
              "until nc -z mysql-service 3306; do echo waiting for mysql; sleep 2; done;",
            ]
        - name: wait-for-redis
          image: busybox
          command:
            [
              "sh",
              "-c",
              "until nc -z redis-service 6379; do echo waiting for redis; sleep 2; done;",
            ]
      containers:
        - name: product-service
          image: meta-web-three/product-service:latest
```

**映射说明**：

- Docker Compose 的 `depends_on` 映射到 Kubernetes 的 `initContainers`
- 使用初始化容器等待依赖服务就绪
- 更精确的依赖控制
- 支持健康检查等待

## 7. 镜像构建

### Docker Compose 构建配置

```yaml
# docker-compose.yml
services:
  client:
    build:
      context: "./client"
      dockerfile: dockerfile

# docker-compose.server.yaml
services:
  product-service:
    build:
      context: ./product-service
      dockerfile: Dockerfile
  user-action-service:
    build:
      context: ./user-action-service
      dockerfile: Dockerfile
```

### Kubernetes 镜像配置

```yaml
# Deployment 中的镜像配置
spec:
  template:
    spec:
      containers:
        - name: product-service
          image: meta-web-three/product-service:latest
        - name: user-action-service
          image: meta-web-three/user-action-service:latest
```

**映射说明**：

- Docker Compose 的 `build` 指令在 Kubernetes 中需要预先构建
- 需要手动构建镜像并推送到镜像仓库
- 建议使用 CI/CD 流水线自动化构建过程

## 8. 服务发现

### Docker Compose 服务发现

```yaml
# 服务间通过服务名直接访问
# 例如：mysql://mysql:3306
```

### Kubernetes 服务发现

```yaml
# Service 提供集群内服务发现
apiVersion: v1
kind: Service
metadata:
  name: mysql-service
spec:
  selector:
    app: mysql
  ports:
    - port: 3306
      targetPort: 3306
  type: ClusterIP
# 应用配置中使用服务名访问
# 例如：mysql://mysql-service:3306
```

**映射说明**：

- Docker Compose 的服务名映射到 Kubernetes 的 Service 名
- Service 提供负载均衡和服务发现
- 支持多种 Service 类型（ClusterIP、NodePort、LoadBalancer）

## 9. 健康检查

### Docker Compose 健康检查

```yaml
# Docker Compose 没有内置健康检查
# 依赖应用自身的健康检查机制
```

### Kubernetes 健康检查

```yaml
# Liveness Probe
spec:
  template:
    spec:
      containers:
        - name: product-service
          livenessProbe:
            httpGet:
              path: /actuator/health
              port: 10082
            initialDelaySeconds: 60
            periodSeconds: 30

          # Readiness Probe
          readinessProbe:
            httpGet:
              path: /actuator/health
              port: 10082
            initialDelaySeconds: 30
            periodSeconds: 10
```

**映射说明**：

- Kubernetes 提供内置的健康检查机制
- Liveness Probe 检测应用是否存活
- Readiness Probe 检测应用是否就绪
- 更可靠的故障检测和恢复

## 10. 资源限制

### Docker Compose 资源限制

```yaml
# Docker Compose 没有内置资源限制
# 依赖 Docker 的资源限制机制
```

### Kubernetes 资源限制

```yaml
# 资源请求和限制
spec:
  template:
    spec:
      containers:
        - name: product-service
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

**映射说明**：

- Kubernetes 提供细粒度的资源管理
- 支持 CPU 和内存的请求和限制
- 更好的资源利用和调度

## 11. 无法直接映射的配置

### 1. Docker Compose 的 `build` 指令

**问题**: Kubernetes 不直接支持构建镜像
**解决方案**:

- 预先构建镜像并推送到镜像仓库
- 使用 CI/CD 流水线自动化构建过程
- 使用 Kaniko 或 BuildKit 在 Kubernetes 中构建

### 2. Docker Compose 的 `container_name`

**问题**: Kubernetes 自动生成 Pod 名称
**解决方案**:

- 使用标签和选择器进行服务发现
- 使用 StatefulSet 获得稳定的网络标识

### 3. Docker Compose 的 `depends_on`

**问题**: Kubernetes 没有直接的依赖关系
**解决方案**:

- 使用 `initContainers` 等待依赖服务就绪
- 使用 Helm 的依赖管理
- 在应用代码中实现重试机制

### 4. Docker Compose 的 `external_links`

**问题**: Kubernetes 不支持外部链接
**解决方案**:

- 使用 Service 和 Endpoints
- 使用 ExternalName Service
- 使用 API Gateway 或 Service Mesh

## 12. 最佳实践建议

### 1. 使用 Helm Chart

```bash
# 创建 Helm Chart
helm create meta-web-three
# 将配置转换为 Helm 模板
```

### 2. 配置管理

- 使用 ConfigMap 管理非敏感配置
- 使用 Secret 管理敏感信息
- 使用外部配置管理系统（如 Vault）

### 3. 存储管理

- 使用 StorageClass 定义存储类型
- 使用 PersistentVolumeClaim 申请存储
- 考虑使用 StatefulSet 管理有状态服务

### 4. 网络管理

- 使用 NetworkPolicy 控制 Pod 间通信
- 使用 Ingress 提供外部访问
- 考虑使用 Service Mesh（如 Istio）

### 5. 监控和日志

- 集成 Prometheus + Grafana 监控
- 使用 ELK Stack 收集日志
- 配置告警和通知

## 总结

Kubernetes 配置相比 Docker Compose 提供了：

- ✅ 更强大的编排能力
- ✅ 更好的可扩展性
- ✅ 更安全的配置管理
- ✅ 更灵活的存储管理
- ✅ 更可靠的故障恢复
- ✅ 更丰富的监控和日志功能

需要额外处理的部分：

- 🔧 镜像构建和推送
- 🔧 存储目录创建
- 🔧 域名和证书配置
- 🔧 监控和日志收集
- 🔧 备份策略
