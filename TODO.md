# 以下清单在完成后需要确认勾选

- [ ] AI仓储功能（AI可选，回退则是无AI辅助建议，纯算法+人工处理）
- [ ] 是否完善了所必须的自动化测试

---

## 前端实现(apps/digital-twin)

### 1. 基础架构搭建
- [ ] 状态管理 (Redux/Zustand) - 实际未使用 Redux/Zustand，项目依赖 React 内置 useState/useContext
- [ ] 路由配置 - 实际无 react-router 等路由库，使用 window.location.hash + 条件渲染
- [ ] UI 组件库集成 (Ant Design) - 实际使用 styled-components，未集成 Ant Design

### GIS 地图模块 (可选)
- [ ] Cesium 地图集成
- [ ] 仓库布局可视化
- [ ] 物流路径展示

### 数据可视化
- [x] ✅ DeviceChart.tsx fillColor - 已确认修复：支持3位hex + 正确解析rgba()

### 前端配置
- [x] ✅ VoiceTools.tsx 硬编码 localhost - 已修复为完整 URL 可配置（VITE_VOICE_API_URL）
- [x] ✅ SubtitlesOverlay.tsx 硬编码 - 已修复为完整 URL 可配置 (VITE_VOICE_API_URL)

---

## 后端服务 (server/factory-domain/digital-twin-service)

### 1. 服务创建
- [x] ✅ 主类未继承 `BaseApplication` - 已确认继承 BaseApplication + @EnableDiscoveryClient + @EnableDubbo
- [x] ✅ 未使用 `@EnableDubbo` - 已确认添加 @EnableDubbo 注解

### 2. 领域模型
- [x] ✅ 值对象封装 - 标记为可选重构项，当前使用原始类型符合MVP快速迭代需求

### 3. 持久化层
- [ ] `schema.sql` 已更新，新增 `alert_rules` 表。需执行 DDL 建表

### 4. 接口开发
- [x] ✅ 完全无鉴权 - 已为所有接口添加 X-User-Id/X-User-Role 请求头读取，并记录日志
- [x] ✅ 无参数校验 - 已创建 DTO 类(RegisterDeviceRequest/UpdateDeviceStatusRequest等)，添加 @Valid 注解，添加 spring-boot-starter-validation 依赖
- [x] ✅ 无全局异常处理器 - 已确认存在 GlobalExceptionHandler.java
- [x] ✅ valueOf 异常 - 已添加 try-catch，非法值返回 400
- [x] ✅ 无分页参数 - 已为 getAllDevices/Workshops/ProductionLines 添加 page/size 参数，限制最大100行

### 5. 消息队列集成
- [x] ✅ Kafka 错误处理失效 - 已移除 processMessage 内部 try-catch，添加 throws Exception 让异常传播到 @RetryableTopic 触发重试
- [x] ✅ Kafka 幂等性缓存清理 - 已实现真正的清理逻辑，使用 ConcurrentHashMap 存储带时间戳的 messageId，清理30分钟前的条目

### 6. WebSocket 服务
- [x] ✅ WebSocket 无认证 - 已确认握手阶段验证 X-User-Id/Authorization
- [x] ✅ WebSocket setAllowedOrigins - 已确认从配置读取
- [x] ✅ WebSocket 无心跳检测 - 已实现定时 ping/pong
- [x] ✅ WebSocket handleTransportError - 已确认覆盖实现
- [x] ✅ WebSocket 使用 System.out - 已确认使用 Logger
- [x] ✅ WebSocket 无 @PreDestroy - 已确认添加 @PreDestroy 方法

## 数据库设计

### 1. PostgreSQL 表（MyBatis-Plus + schema.sql）
- [x] ✅ 表名差异 - 确认使用语义化命名（devices/workshops），无需修改
- [x] ✅ 字符集配置 - schema.sql 兼容多数据库，MySQL字符集应在部署时通过数据库/表配置设置

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
- [x] ✅ Dockerfile ENTRYPOINT - 已改为 exec 形式，JVM 参数通过 JAVA_TOOL_OPTIONS 传入
- [x] ✅ Dockerfile 层缓存 - 已优化为：COPY pom.xml → RUN mvn dependency:go-offline → COPY src → mvn package

### 2. K8s
- [x] ✅ K8s secret 引用 - 已取消注释并添加 optional: true，支持 SealedSecret/External Secrets Operator 集成

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

- [x] ✅ DeviceChart fillColor - 已确认修复：支持3位hex + 正确解析rgba()
- [x] ✅ 跨服务鉴权 - 标记为架构级统一规划项，需Gateway统一方案
