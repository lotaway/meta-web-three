# 以下清单在完成后需要确认勾选

- [ ] AI仓储功能（AI可选，回退则是无AI辅助建议，纯算法+人工处理）
- [ ] 所有docker和k8s运维配置文件是否都对应了当前全新的项目架构
- [ ] 是否完善了所必须的自动化测试

---

# 数字孪生 (Digital Twin) 实现清单

## 技术栈确认

- [x] 3D 可视化: Three.js + react-three/fiber + react-three/drei
- [x] GIS 地图: Cesium
- [x] 实时数据: WebSocket
- [x] IoT 设备: MQTT
- [x] 流数据: Kafka
- [x] 时序数据库: InfluxDB
- [x] 告警: Prometheus
- [x] AI 分析: Python
- [x] 设备连接: OPC-UA

---

## 前端实现(apps/digital-twin) [https://github.com/lotaway/meta-not](以此为基础开始增删工作)

### 1. 基础架构搭建
- [x] 项目初始化 (React + TypeScript) - 已使用 meta-note 脚手架
- [x] 状态管理 (Redux/Zustand) - 已存在于 meta-note
- [x] 路由配置 - 已存在于 meta-note
- [x] UI 组件库集成 (Ant Design) - 已存在于 meta-note

### 2. 3D 场景模块
- [x] Three.js 场景初始化 - FactoryScene.tsx
- [x] 工厂车间 3D 模型加载 - Grid + Floor
- [x] 设备 3D 模型展示 - DeviceModel 组件
- [x] AGV/机器人 3D 模型动画 - 基础模型
- [x] 相机控制 (旋转、缩放、平移) - OrbitControls
- [x] 场景灯光与环境配置 - 环境光 + 平行光

### 3. 实时数据展示
- [x] WebSocket 客户端配置 - websocket.ts
- [x] 设备状态实时更新 - DeviceStatus.tsx
- [x] 生产线节拍可视化 - DigitalTwinPage
- [x] 产量/效率实时看板 - DeviceChart.tsx + StatsCard

### 4. GIS 地图模块 (可选)
- [ ] Cesium 地图集成
- [ ] 仓库布局可视化
- [ ] 物流路径展示

### 5. 数据可视化
- [ ] ECharts 图表集成
- [ ] 产量趋势图
- [ ] 设备 OEE 仪表盘
- [ ] 告警统计饼图

### 6. IoT 设备接入
- [ ] MQTT 客户端配置
- [ ] 设备状态主题订阅
- [ ] 传感器数据解析
- [ ] 设备告警推送

### 7. 告警管理
- [ ] 告警列表展示
- [ ] 告警规则配置
- [ ] 告警等级分类 (红/黄/蓝)
- [ ] 告警声音/弹窗通知

---

## 后端服务 (server/factory-domain/digital-twin-service)

### 1. 服务创建
- [x] digital-twin-service 目录结构 (DDD 架构)
- [x] pom.xml 依赖配置
- [x] Application 启动类

### 2. 领域模型
- [x] DeviceEntity - 设备实体
- [x] WorkshopEntity - 车间实体
- [x] ProductionLineEntity - 生产线实体
- [x] AlertEntity - 告警实体

### 3. 接口开发
- [x] 设备列表查询 API
- [x] 设备实时状态 API
- [x] 历史数据查询 API
- [x] 告警记录 API

### 4. 消息队列集成
- [x] Kafka Consumer 配置
- [x] 设备状态消息消费
- [x] 事件转发到 WebSocket

### 5. WebSocket 服务
- [x] WebSocket 配置
- [x] 客户端连接管理
- [x] 实时数据推送

---

## 数据库设计

### 1. 持久化迁移（当前全部使用 ConcurrentHashMap 内存存储，宕机即丢）

> **注意：** `server/common/` 已通过 `application-common.yml` 提供了以下公共配置，digital-twin-service 无需重复定义：
> - PostgreSQL 数据源（`spring.datasource`）
> - MyBatis-Plus 全局配置（驼峰映射、分页插件、自动填充、ID 策略）
> - Redis 连接（`spring.data.redis`）
> - Elasticsearch 连接
>
> digital-twin-service 的 `pom.xml` 已依赖 `common` 模块，只需将自身的 `application.yml` 配置集成 `application-common.yml` 即可（`spring.config.import: classpath:application-common.yml`）。

- [ ] 在 `application.yml` 中添加 `spring.config.import: classpath:application-common.yml`，继承 common 的公共数据库配置
- [ ] 在 `digital-twin-service` 领域实体上添加 MyBatis-Plus 注解（`@TableName`、`@TableId` 等），映射到 `schema.sql` 对应的表
- [ ] 删除 `DeviceRepositoryImpl.java` 及其接口，替换为 MyBatis-Plus `BaseMapper<Device>`
- [ ] 删除 `WorkshopRepositoryImpl.java` 及其接口，替换为 MyBatis-Plus `BaseMapper<Workshop>`
- [ ] 删除 `ProductionLineRepositoryImpl.java` 及其接口，替换为 MyBatis-Plus `BaseMapper<ProductionLine>`
- [ ] 删除 `AlertRepositoryImpl.java` 及其接口，替换为 MyBatis-Plus `BaseMapper<Alert>`
- [ ] 删除或重构 `schema.sql`（执行方式改为 Flyway 或 MyBatis-Plus `ddl-auto`）
- [ ] 重构 `DigitalTwinDevDataInitializer` — 种子数据改为基于数据库状态（而非 `ConcurrentHashMap` 判空）

### 2. 时序数据存储（抽象接口，支持多后端切换）

> 通过定义统一接口，PostgreSQL 和 InfluxDB 可共存切换，适配不同客户需求：
> - **小/中型**：PostgreSQL 实现（零额外运维，SQL 通用）
> - **大型/高频**：InfluxDB 实现（高性能压缩+自动降采样）
>
> Spring Profile 或配置项控制 `@ConditionalOnProperty`，切换后端无需改业务代码

#### 公共接口抽象
- [ ] 定义时序数据写入接口 `TelemetryRepository`（`saveTelemetry()`, `saveProductionLog()`, `queryHistory()` 等）
- [ ] 定义时序数据查询接口（范围查询、聚合查询、最新值查询）

#### PostgreSQL 实现（默认，小中型场景）
- [ ] 创建 `ProductionLog` 和 `TelemetryLog` 的 MyBatis-Plus Entity + Mapper，对应 `schema.sql` 的时序表
- [ ] 实现 `PostgresTelemetryRepository`，写入 PostgreSQL
- [ ] 实现时序数据查询 API（支持时间范围过滤、聚合，供前端 ECharts 趋势图使用）
- [ ] `DigitalTwinSimulator` 模拟数据改为同时写入时序表

#### InfluxDB 实现（可选，大型高频场景）
- [ ] 保留 `pom.xml` 中的 `influxdb-client-java` 依赖
- [ ] 在 `docker-compose.env.yml` 添加 InfluxDB 服务（可选，按需启用）
- [ ] 配置 InfluxDB 连接（`application-common.yml` 或独立配置）
- [ ] 实现 `InfluxdbTelemetryRepository`，写入 InfluxDB
- [ ] 实现数据生命周期管理（Retention Policy + 自动降采样）

### 3. Redis 实时缓存（common 已提供 Redis 连接配置，digital-twin-service 无需重复配置）

- [ ] 使用 common 提供的 Redis 连接，在 digital-twin-service 中缓存设备最新状态（替代内存 Map，实现进程重启不丢）
- [ ] 使用 Redis Pub/Sub 作为 WebSocket 广播的背板（多实例场景）

### 4. 已有 DDL 但未实现的表（对应 `schema.sql`）

- [ ] `workshops` - 车间配置表（已有 DDL，数据库表中无数据，需创建 MyBatis-Plus Entity）
- [ ] `production_lines` - 生产线表（已有 DDL，数据库表中无数据，需创建 MyBatis-Plus Entity）
- [ ] `devices` - 设备信息表（已有 DDL，数据库表中无数据，需创建 MyBatis-Plus Entity）
- [ ] `alerts` - 告警记录表（已有 DDL，数据库表中无数据，需创建 MyBatis-Plus Entity）
- [ ] `production_logs` - 产量日志表（时序数据，建在 PostgreSQL，见上方时序数据章节）
- [ ] `telemetry_logs` - 遥测日志表（时序数据，建在 PostgreSQL，见上方时序数据章节）

---

## 联调测试

### 1. 单元测试
- [ ] 领域模型单元测试
- [ ] Service 层单元测试

### 2. 集成测试
- [ ] WebSocket 集成测试
- [ ] MQTT 消息集成测试
- [ ] Kafka 消费集成测试

### 3. 端到端测试
- [ ] 3D 场景加载测试
- [ ] 实时数据展示测试
- [ ] 告警流程测试

---

## 部署配置

### 1. Docker
- [ ] digital-twin-service/Dockerfile
- [ ] docker-compose.digital-twin.yml

### 2. K8s
- [ ] k8s/digital-twin-deployment.yaml
- [ ] k8s/digital-twin-service.yaml

---

## 生产环境缺陷清单（待修复）

- [ ] system-management 的 TTS 仍为 mock：`apps/digital-twin/system-management/src/main/nestjs/services/tts.service.ts` 中 `synthesize()` 返回 `Buffer.from('mock-audio-data')`（需要替换真实合成逻辑，未就绪时应 fail-fast/清晰报错并让前端不可用）
- [ ] system-management 的 native 加载失败后使用占位降级：`apps/digital-twin/system-management/src/main/rust-bridge.ts` catch 分支里替换 native 为 `startCaptureService` 返回字符串“Native module not loaded”、`stopCaptureService` 返回 false（需要改为严格失败/可观测告警/让 UI 进入不可用状态）
- [ ] system-management 的 WebSocket 存在示例式占位回复（需替换为正式协议）：`apps/digital-twin/system-management/src/main/nestjs/services/websocket.service.ts` 中包含“External login no longer supported”“Hello from ${appProtocol}”（建议替换为标准握手/错误码流程）

