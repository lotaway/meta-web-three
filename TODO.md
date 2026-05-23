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
- [ ] AGV/机器人 3D 模型动画 - 只有静态几何体（box/cylinder），无 useFrame/useSpring/lerp 插值，位置跳变。需实现：AGV 移动插值、ROBOT 旋转动画
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
- [x] ECharts 图表集成 - ECharts.tsx
- [x] 产量趋势图 - LineChart
- [x] 设备 OEE 仪表盘 - GaugeChart
- [x] 告警统计饼图 - PieChart
- [x] 设备实时图表 - DeviceChart.tsx
- [ ] ⚠️ DeviceChart.tsx:101 fillStyle 颜色格式 Bug：`color.replace(')', ', 0.1)')` 在 hex 颜色上不匹配（无 `)`），导致 area fill 回退黑色，需修复颜色转换逻辑

### 6. IoT 设备接入
- [x] MQTT 客户端配置 - mqtt.ts
- [x] 设备状态主题订阅 - useDigitalTwinMQTT
- [x] 传感器数据解析 - DeviceTelemetry
- [x] 设备告警推送 - AlertPanel
- [ ] ⚠️ MQTT hook 依赖数组仅追踪 `options.brokerUrl`（mqtt.ts:160），topics/密码变更不会触发重连
- [ ] ⚠️ MQTT connect 失败（mqtt.ts:154）仅 `console.error` 不传播到 hook 的 error 状态

### 7. 告警管理
- [x] 告警列表展示 - AlertPanel.tsx
- [ ] 告警规则配置
- [x] 告警等级分类 (红/黄/蓝) - levelColors
- [ ] 告警声音/弹窗通知
- [ ] ⚠️ WebSocket 断线重连（websocket.ts:84）setTimeout 无 timer ID 追踪，disconnect 后可能触发意外重连

### 8. 前端配置
- [ ] ⚠️ `digital-twin.ts` 默认端口 `localhost:10102` 硬编码，`DigitalTwinPage.tsx:181` UI 文本中也写死了端口号
- [ ] ⚠️ `AudioContext.tsx` 和 `VoiceTools.tsx` 中也有类似的硬编码 localhost URL

---

## 后端服务 (server/factory-domain/digital-twin-service)

### 1. 服务创建
- [x] digital-twin-service 目录结构 (DDD 架构)
- [x] pom.xml 依赖配置
- [x] Application 启动类
<<<<<<< HEAD
- [ ] ❌ 主类未继承 `BaseApplication`，未使用 `@EnableDiscoveryClient`，不向 ZooKeeper 注册
- [ ] ❌ 未使用 `@EnableDubbo`，不支持 RPC

### 2. 领域模型
- [x] Device - 设备实体（有业务行为方法）
- [x] Workshop - 车间实体
- [x] ProductionLine - 生产线实体
- [x] Alert - 告警实体（状态机完整）
- [x] AlertRule - 告警规则实体
- [ ] ⚠️ 存在贫血 setter（如 `Device.setStatus()` 绕过 `goOnline()` 等方法），建议移除公有 setter 或加注解标记
- [ ] ⚠️ 缺少 `equals()`/`hashCode()`，基于 ID 的实体等同性无法保证
- [ ] ⚠️ 未使用值对象封装（`WorkshopId`、`Position`、`IPAddress` 等原始类型）

### 3. 持久化层
- [x] ~~内存 ConcurrentHashMap 存储~~ → 已替换为 MyBatis-Plus + PostgreSQL
- [x] `MybatisPlusConfig` 继承 common 的 `MybatisPlusDefaultConfig`
- [x] 5 个 DO 对象（`@TableName` 注解，继承 `BaseDO` 自动填充时间戳）
- [x] 5 个 Mapper（继承 `BaseMapper<T>`）
- [x] 5 个 Converter（`toDO()` / `toEntity()` 双向转换）
- [x] 5 个 RepositoryImpl（使用 Mapper 替代 ConcurrentHashMap）
- [ ] `schema.sql` 已更新，新增 `alert_rules` 表。需执行 DDL 建表

### 4. 接口开发
- [x] 设备列表查询 API
- [x] 设备实时状态 API
- [x] 历史数据查询 API
- [x] 告警记录 API
- [x] 告警规则管理 API（AlertRuleController）
- [ ] ❌ **完全无鉴权**：全控制器无 `@RequirePermission`，未读取 `X-User-Id`/`X-User-Role` 头
- [ ] ❌ 无参数校验：全部使用 `Map<String, Object>` 而非 DTO，无 `@Valid`/`@Validated`
- [ ] ❌ 无全局异常处理器（`@RestControllerAdvice`）
- [ ] ❌ `valueOf(status.toUpperCase())` 传入非法值会抛出未捕获异常导致 500
- [ ] ❌ 无分页参数、无限流机制
- [ ] ⚠️ 路径命名不统一（单数 `/device/{id}` vs 复数 `/devices`）

### 5. 消息队列集成
- [x] Kafka Consumer 监听器
- [x] 6 个 Topic：`device.status.changed`、`device.position.updated` 等
- [ ] ❌ **`DigitalTwinEventPublisher` 是伪发布器**：publish 方法只打日志未实际发送 Kafka 消息
- [ ] ❌ 无错误处理：Kafka 监听器无 try-catch，`broadcast()` 异常会导致消费失败
- [ ] ❌ 无重试机制（`@RetryableTopic`）、无死信队列（DLT）
- [ ] ⚠️ 无幂等性处理，位置更新等消息重复消费无保护

### 6. WebSocket 服务
- [x] WebSocket 配置 `/ws/digital-twin`
- [x] 客户端连接管理（`ConcurrentHashMap.newKeySet()`）
- [x] 实时数据推送（`broadcast()`）
- [ ] ❌ **完全无认证**：`afterConnectionEstablished` 直接放行
- [ ] ❌ **`setAllowedOrigins("*")`** 全源允许，生产环境需限制具体域名
- [ ] ❌ 无心跳检测（ping/pong）
- [ ] ❌ 无 `handleTransportError` 覆盖，传输异常不清理 session
- [ ] ❌ 使用 `System.out.println` 和 `e.printStackTrace()`，应替换为 Logger
- [ ] ❌ 无 `@PreDestroy` 优雅关闭
=======

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
>>>>>>> c8922596ee67e5c02c23dcd594e8b0476339df66

---

## 数据库设计

<<<<<<< HEAD
### 1. PostgreSQL 表（MyBatis-Plus + schema.sql）
- [x] workshops（车间表）- schema.sql
- [x] devices（设备表）- schema.sql
- [x] production_lines（产线表）- schema.sql
- [x] alerts（告警记录表）- schema.sql
- [x] alert_rules（告警规则表）- schema.sql（已补充）
- [ ] ⚠️ 表名与 TODO 有差异（`devices` 非 `device_info`，`workshops` 非 `workshop_config` 等）
- [ ] ⚠️ `data.sql` 含 PostgreSQL 专有语法（如 `CURRENT_TIMESTAMP - INTERVAL '1' HOUR`），与 schema.sql 兼容 H2/MySQL 的定位矛盾
- [ ] ⚠️ CREATE TABLE 未指定字符集（MySQL 场景下中文可能出问题）
- [ ] ⚠️ 外键约束缺失（`production_lines.workshop_id`、`devices.workshop_id` 等无 FK）
=======
### 1. 持久化迁移（当前全部使用 ConcurrentHashMap 内存存储，宕机即丢）
>>>>>>> c8922596ee67e5c02c23dcd594e8b0476339df66

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
- [x] digital-twin-service/Dockerfile（多阶段构建、非 root、健康检查 ✅）
- [x] docker-compose.digital-twin.yml（端口映射、依赖控制 ✅）
- [ ] ⚠️ Dockerfile `ENTRYPOINT` 使用 `sh -c` 包装 JVM，`docker stop` 信号无法直接传递到 JVM 进程，需改用 `exec` 形式
- [ ] ⚠️ Dockerfile 未利用层缓存（应先 COPY pom.xml 下载依赖，再 COPY 源码）

### 2. K8s
- [x] k8s/digital-twin-deployment.yaml（副本数、资源限制、探针、反亲和性 ✅）
- [x] k8s/digital-twin-service.yaml（ClusterIP + LoadBalancer ✅）
- [x] k8s/digital-twin/ingress.yaml（TLS + WebSocket 支持 ✅）
- [x] k8s/digital-twin/ingress.yaml
- [ ] ❌ K8s deployment 环境变量硬编码（secret 引用缺失），需迁移到 `SealedSecret`/External Secrets Operator
- [ ] ❌ **`.env` 文件已提交到 Git**，包含 `MYSQL_ROOT_PASSWORD=123123`、`REDIS_PASSWORD=123123` 等弱密码，需立即移除并刷新凭据
- [ ] ⚠️ 镜像拉取策略 `Always`，应使用具体版本标签

---

## 跨服务鉴权体系延伸（架构级，须统一处理）

Gateway（`server/gateway`）的 `UserAuthFilter` + JWT 鉴权体系原本只服务商城（mall-domain），现有 digital-twin、ERP、供应链等新域均未纳入。

### 当前状态

| 服务 | 注册到 ZooKeeper？ | Gateway 自动路由？ | 消费 `X-User-Id`/`X-User-Role` 头？ | 使用 `@RequirePermission`？ |
|------|-------------------|-------------------|--------------------------------------|----------------------------|
| `user-service` | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| `product-service` | ✅ 是 | ✅ 是 | ⚠️ 不消费但 Gateway 已验证 JWT | ⚠️ 无 |
| `order-service` | ✅ 是 | ✅ 是 | ⚠️ 同上 | ⚠️ 无 |
| **`digital-twin-service`** | ❌ 否（未继承 `BaseApplication`） | ❌ 否（ZooKeeper 不可发现） | ❌ 完全不读取 | ❌ 无 |
| **ERP 各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |
| **供应链各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |

### 需要做的工作（跨域统一方案）

1. **digital-twin-service**：主类继承 `BaseApplication` + `@EnableDiscoveryClient` + `@EnableDubbo`，注册到 ZooKeeper
2. **Gateway 白名单**：若 digital-twin WebSocket 端点无需鉴权，需加入 `excludedPathPatterns`；其余 API 通过 Gateway 统一鉴权
3. **Controller 改造**：注入 `@RequestHeader(HeaderConstants.USER_ID)` 和 `@RequestHeader(HeaderConstants.USER_ROLE)`，在业务方法中做授权决策
4. **`@RequirePermission` 拓展**：当前仅 admin 权限体系，需扩充到 digital-twin 的 device/workshop/alert 等资源（如 `dt:device:control`、`dt:alert:ack`）
5. **WebSocket 认证**：在握手阶段通过 `HandshakeInterceptor` 验证 token（query param 或 Sec-WebSocket-Protocol）
6. **ERP + 供应链同理**：统一规划权限资源树，避免各域各自造轮子

---

## 生产环境缺陷清单（待修复）

- [x] system-management 的 TTS 仍为 mock：已替换为真实系统 TTS 合成（Rust native 模块 `synthesize_speech()` 调用 macOS `say`/Windows SAPI/Linux espeak，输出 PCM WAV）；native 模块不可用时 fail-fast + 前端不可用
- [x] system-management 的 native 加载失败后使用占位降级：已删除 catch 分支中的 mock 对象，`native = null`，所有消费者通过 `?.`/`??` 安全处理 null
- [x] system-management 的 WebSocket 存在示例式占位回复：已替换为标准 JSON 协议（`hello_ack`/`pong`/`subscribe_ack`/`unsubscribe_ack`，错误码 4000/4001）
- [x] **后端持久化**：5 个 RepositoryImpl 从 ConcurrentHashMap 内存存储迁移到 MyBatis-Plus + PostgreSQL（DO/Mapper/Converter/MybatisPlusConfig 全部就绪）
- [ ] ⚠️ AGV/机器人 3D 模型动画标记为 [x] 但实际未实现（仅有静态几何体），已更正为 [ ]
- [ ] ⚠️ DeviceChart.tsx:101 fillStyle 颜色转换 Bug（hex → rgba 格式错误，area fill 不生效）
- [ ] ⚠️ `DigitalTwinEventPublisher` 只打印日志不发送 Kafka 消息，属"伪发布器"
- [ ] ⚠️ 跨服务鉴权：digital-twin + ERP + 供应链均未纳入 Gateway 鉴权体系，见下方"跨服务鉴权体系延伸"章节

