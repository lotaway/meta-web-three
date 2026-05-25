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
- [x] 领域模型单元测试
- [x] DigitalTwinDomainServiceImpl 核心领域服务测试
- [x] DigitalTwinCommandService 命令服务测试
- [x] DigitalTwinQueryService 查询服务测试
- [x] DigitalTwinKafkaConsumer Kafka消费核心测试
- [x] DigitalTwinWebSocketHandler WebSocket核心测试
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

### 🟡 严重级 (High) — CODE_PRICEPLES 违规


- [x] **注释残留 — 清理非必要注释（已修复）**
  - 位置: 多处文件（KafkaConsumer, DeviceChart.tsx 等）
  - 问题: 违反"禁止注释，任何需要注释解释的代码视为设计失败"
  - 方案: 删除功能实现注释，将需解释逻辑重构为自描述代码
  - **已修复**: `DigitalTwinKafkaConsumer.java` 清理 L36/L49/L75 等注释；`DeviceChart.tsx` 清理全文件数十处注释

- [x] **单函数超 20 行 — 拆分大函数（已修复）**
  - 违反清单:
    - `DigitalTwinKafkaConsumer.processMessage()` ~35行 → 已拆分为独立 handler 方法 ✅
    - `DeviceChart.tsx fillColor()` ~30行 → 已抽离为独立工具函数 toFillColor() ✅
    - `AlertRuleCommandService.toResponse()` ~24行 → 已使用 enumToString/toStringOrNull 辅助方法 ✅
    - `DigitalTwinSimulator.simulateAGVMovement()` ~23行 → 已拆解为 computeNextIndex/parseCoordinates/computeRotation ✅
  - **已修复**: 每个函数不超过 20 行，一个函数只做一件事

- [x] **核心业务逻辑缺少单元测试（已创建测试文件，阻塞于项目 proto 编译问题）**
  - 已创建测试:
    - `DigitalTwinDomainServiceImplTest` — 核心领域服务测试 ✅
    - `DigitalTwinCommandServiceTest` — 命令服务测试 ✅
    - `DigitalTwinQueryServiceTest` — 查询服务测试 ✅
    - `DigitalTwinKafkaConsumerTest` — Kafka消费核心测试 ✅
    - `DigitalTwinWebSocketHandlerTest` — WebSocket核心测试 ✅
  - **状态**: 测试文件已创建（使用 JUnit 5 + Mockito），共计 ~874 行测试代码
  - **阻塞原因**: 项目 proto 配置文件存在多处 import 路径问题，导致 common 模块无法编译，已修复 `UserRiskProfileService.proto` 和 `RiskScorerService.proto` 的 DeviceRiskTag import，但仍有 google/type/money.proto、web3/token.proto 等多处问题

### 🔵 一般级 (Medium)

- [x] **`DigitalTwinEventPublisher` 重复 Logger 定义（已修复）**
  - 位置: `DigitalTwinEventPublisher.java:17`
  - 问题: 删除 `@Slf4j` 后未添加 `import org.slf4j.Logger` 和 `import org.slf4j.LoggerFactory`，导致编译错误
  - **已修复**: 补全了缺失的 `import org.slf4j.Logger;` 和 `import org.slf4j.LoggerFactory;` 导入语句

- [x] **硬编码 CORS 来源改为配置注入**
  - 位置: `DigitalTwinController.java:17`
  - 问题: `@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})` 硬编码
  - 方案: 从 `application.yml` 读取 `digital-twin.cors.allowed-origins` 配置注入
  - **已修复**: 创建 `DigitalTwinCorsConfig.java` 配置类，从配置读取 allowed-origins 数组，通过 `WebMvcConfigurer` 配置全局 CORS

- [x] **内存分页改为数据库分页**
  - 位置: `DigitalTwinQueryService.java`
  - 问题: 三处分页均使用 `findAll()` + `subList()` 全量加载到内存再截取
  - 方案: 使用 MyBatis-Plus Page 或 SQL `LIMIT/OFFSET` 实现数据库分页
  - **已修复**: 在 `DeviceRepository`、`WorkshopRepository`、`ProductionLineRepository` 添加 `findPaginated(int page, int size)` 方法，使用 `IPage<>` 实现数据库分页，QueryService 改用数据库分页

