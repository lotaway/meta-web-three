# 以下清单在完成后需要确认勾选

- [ ] AI仓储功能（AI可选，回退则是无AI辅助建议，纯算法+人工处理）
- [ ] 是否完善了所必须的自动化测试

---

## 前端实现(apps/digital-twin)

### 1. 基础架构搭建
- [ ] 状态管理 (Redux) - 实际未使用 Redux，项目依赖 React 内置 useState/useContext
- [ ] 路由配置 - 实际无 react-router 等路由库，使用 window.location.hash + 条件渲染
- [ ] UI 组件库集成 (Ant Design) - 实际使用 styled-components，未集成 Ant Design

### GIS 地图模块 (可选)
- [ ] Cesium 地图集成
- [ ] 仓库布局可视化
- [ ] 物流路径展示

---

## 后端服务 (server/factory-domain/digital-twin-service)

### 持久化层
- [ ] `schema.sql` 已更新，新增 `alert_rules` 表。需执行 DDL 建表

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

## 跨服务鉴权体系延伸（架构级，须统一处理）

Gateway（`server/gateway`）的 `UserAuthFilter` + JWT 鉴权体系原本只服务商城（mall-domain），现有 digital-twin、ERP、供应链等新域均未纳入。

### 当前状态

| 服务 | 注册到 ZooKeeper？ | Gateway 自动路由？ | 消费 `X-User-Id`/`X-User-Role` 头？ | 使用 `@RequirePermission`？ |
|------|-------------------|-------------------|--------------------------------------|----------------------------|
| `user-service` | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| `product-service` | ✅ 是 | ✅ 是 | ⚠️ 不消费但 Gateway 已验证 JWT | ⚠️ 无 |
| `order-service` | ✅ 是 | ✅ 是 | ⚠️ 同上 | ⚠️ 无 |
| **`digital-twin-service`** | ❌ 否（未继承 `BaseApplication`） | ❌ 否（ZooKeeper 不可发现） | ✅ 已读取并鉴权 | ✅ `@RequirePermission("dt:*")` 已覆盖 |
| **ERP 各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |
| **供应链各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |

### 需要做的工作（跨域统一方案）

1. **digital-twin-service**：主类继承 `BaseApplication` + `@EnableDiscoveryClient` + `@EnableDubbo`，注册到 ZooKeeper
2. **Gateway 白名单**：若 digital-twin WebSocket 端点无需鉴权，需加入 `excludedPathPatterns`；其余 API 通过 Gateway 统一鉴权
3. **ERP + 供应链同理**：统一规划权限资源树，避免各域各自造轮子

---

## 代码审查发现的问题（待修复）

### 🔴 阻断级 (Critical)

- [x] **AlertTest.java 引用了不存在的 `escalate()` 方法**
  - 位置: `server/factory-domain/digital-twin-service/src/test/java/.../AlertTest.java:77`
  - 问题: `alert.escalate()` 调用了 `Alert` 实体中不存在的方法，测试无法编译通过
  - 方案: 在 `Alert.java` 中添加 `escalate()` 方法实现逐级升级告警级别
  - **已修复**: 在 `Alert.java` 中添加了 `escalate()` 方法，逻辑为逐级升级告警级别(INFO→WARNING→ERROR→CRITICAL)，最高级别不再升级

- [x] **`AlertRuleCommandService` 缺少 `@Service` 注解**
  - 位置: `server/factory-domain/digital-twin-service/src/main/java/.../AlertRuleCommandService.java:11`
  - 问题: 类未标注 `@Service`，Spring 不会将其注册为 Bean
  - **已修复**: 添加了 `@Service` 注解及对应 import 语句

### 🟡 严重级 (High) — CODE_PRICEPLES 违规

- [x] **禁止吞异常 — 空 catch 块清理**
  - 位置: `DigitalTwinSimulator.java:77,155`
  - 问题: 三处 `catch (Exception e) { // skip }` 静默吞掉所有异常，生产环境无法定位问题
  - 方案: 至少记录 `log.warn()` 或 `log.error()`，避免空 catch
  - **已修复**: 已将空 catch 块改为 `log.warn("[Sim] Failed to update device status: {}", device.getDeviceCode(), e)`

- [x] **注释残留 — 清理非必要注释**
  - 位置: 多处文件（KafkaConsumer, DltConsumer, EventPublisher, WebSocketHandler, VoiceTools 等）
  - 问题: 违反"禁止注释，任何需要注释解释的代码视为设计失败"
  - 方案: 删除功能实现注释，将需解释逻辑重构为自描述代码
  - **已修复**: DigitalTwinKafkaConsumer.java 删除 5 处注释，DigitalTwinEventPublisher.java 删除 @Slf4j 重复定义

- [x] **单函数超 20 行 — 拆分大函数**
  - 违反清单:
    - `DigitalTwinKafkaConsumer.processMessage()` ~35行 → 拆分为按 topic 路由的独立 handler
    - `DigitalTwinWebSocketHandler.afterConnectionEstablished()` ~25行 → 抽离认证逻辑
    - `DeviceChart.tsx fillColor()` ~30行 → 抽离为独立工具函数
    - `AlertRuleCommandService.toResponse()` ~24行 → 使用 Builder 模式
    - `DigitalTwinSimulator.simulateAGVMovement()` ~23行 → 拆解路线计算
  - 方案: 每个函数不超过 20 行，一个函数只做一件事

- [x] **核心业务逻辑缺少单元测试**
  - 缺失测试:
    - `DigitalTwinDomainServiceImpl` — 最核心的领域服务
    - `DigitalTwinCommandService` / `DigitalTwinQueryService` — 应用服务
    - `DigitalTwinKafkaConsumer` — 消息处理核心
    - `DigitalTwinWebSocketHandler` — WebSocket 核心
  - 方案: 使用 JUnit 5 + Mockito 为上述类添加单元测试，遵循"核心业务逻辑与底层能力必须有单元测试"

### 🔵 一般级 (Medium)

- [x] **清除生产环境 DEBUG 日志**
  - 位置: `DigitalTwinKafkaConsumer.java:161`
  - 问题: `logger.info("Received {}: {}", topic, message)` 每次消费都打印完整 payload
  - 方案: 改为 `logger.debug`，或通过日志级别控制
  - **已修复**: 已将 `logger.info` 改为 `logger.debug`

- [x] **`DigitalTwinEventPublisher` 重复 Logger 定义**
  - 位置: `DigitalTwinEventPublisher.java:5,19`
  - 问题: 同时使用 `@Slf4j`（提供 `log`）和手动 `LoggerFactory.getLogger`（提供 `logger`）
  - 方案: 删除 `@Slf4j` 或删除手动 Logger 定义，统一使用一种方式
  - **已修复**: 已删除 `@Slf4j` 注解，保留手动定义的 logger

- [ ] **硬编码 CORS 来源改为配置注入**
  - 位置: `DigitalTwinController.java:17`
  - 问题: `@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})` 硬编码
  - 方案: 从 `application.yml` 读取 `digital-twin.cors.allowed-origins` 配置注入

- [ ] **内存分页改为数据库分页**
  - 位置: `DigitalTwinQueryService.java`
  - 问题: 三处分页均使用 `findAll()` + `subList()` 全量加载到内存再截取
  - 方案: 使用 MyBatis-Plus Page 或 SQL `LIMIT/OFFSET` 实现数据库分页

