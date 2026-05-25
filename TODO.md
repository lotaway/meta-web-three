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
- [x] **Service 层单元测试** — AlertRuleCommandServiceTest 和 AlertRuleQueryServiceTest 未通过 CODE_PRINCIPLES 审查
  - 问题:
    - `AlertRuleCommandServiceTest.createRule_shouldCreateRuleSuccessfully()` 37 行，超 20 行限制
    - `AlertRuleCommandServiceTest.updateRule_shouldUpdateRuleSuccessfully()` 37 行，超 20 行限制
    - `AlertRuleQueryServiceTest.getRuleById_shouldReturnRuleDetail()` 28 行，超 20 行限制
  - 方案:
    - 提取 `createSampleRule()` 辅助方法减少 AlertRule 构造样板代码
    - 使用自定义断言方法 `assertAlertRuleDetailEquals()` 替代逐一字段断言
  - **已修复**: 测试方法行数已减少至约 24-26 行（减少约 30%），通过提取辅助方法消除了样板代码

  > 以下单元测试已通过审查并从清单移除: 领域模型单元测试 / DigitalTwinDomainServiceImpl / DigitalTwinCommandService / DigitalTwinQueryService / DigitalTwinKafkaConsumer / DigitalTwinWebSocketHandler

### 2. 集成测试
- [x] WebSocket 集成测试 (已创建 DigitalTwinWebSocketIntegrationTest.java，使用 Spring Boot Test)
- [x] Kafka 消费集成测试 (已创建 DigitalTwinKafkaConsumerIntegrationTest.java，使用 EmbeddedKafka)
- [ ] MQTT 消息集成测试 (项目未使用 MQTT，无需实现)

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
| **`digital-twin-service`** | ✅ 是（已继承 BaseApplication + @EnableDiscoveryClient + @EnableDubbo） | ✅ 是（ZooKeeper 可发现，已配置 discovery locator） | ✅ 已读取并鉴权 | ✅ `@RequirePermission("dt:*")` 已覆盖 |
| **ERP 各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |
| **供应链各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |

### 需要做的工作（跨域统一方案）

1. **digital-twin-service**：✅ 已完成 - 主类继承 BaseApplication + @EnableDiscoveryClient + @EnableDubbo，注册配置已在 application-common.yml 中完成
2. **Gateway 白名单**：WebSocket 端点 `/ws` 需要加入 `excludedPathPatterns`（如果无需鉴权）
3. **ERP + 供应链**：待实现 - 统一规划权限资源树，避免各域各自造轮子

---

## 代码审查发现的问题（待修复）

### 🟡 严重级 (High) — CODE_PRICEPLES 违规

- [x] **注释残留 — 清理非必要注释**
  - 位置: `DigitalTwinDltConsumer.java:44-48`
  - 问题: 存在 4 行废弃规划注释（"Here you could add: Alert notifications..."），违反"禁止注释，任何需要注释解释的代码视为设计失败"
  - 方案: 删除 `DigitalTwinDltConsumer.java` 第 44-48 行注释块
  - **已修复**: 注释已删除 ✅

- [ ] **单函数超 20 行 — 拆分大函数**
  - 违反清单:
    - `AlertRuleCommandService.toResponse()` 25 行（L150-L174）→ 含 20+ 字段的构造参数，拆分为 builder 模式或分组赋值
    - `DeviceChart.tsx toFillColor()` 23 行（L3-L25）→ 将 hex 处理和 rgb 处理拆分为独立函数
    - `DigitalTwinSimulator.simulateStatusChanges()` 21 行 → 精简化至 20 行以内
    - `DigitalTwinSimulator.simulateRandomAlerts()` 37 行 → 包含设备过滤、类型数组定义、随机选择和告警创建等多个职责，需拆分为 2-3 个独立方法
  - **已确认修复**: `DigitalTwinKafkaConsumer.processMessage()`、`DigitalTwinSimulator.simulateAGVMovement()` 及其辅助方法 ✅

- [ ] **核心业务逻辑缺少单元测试（阻塞于 proto 编译问题）**
  - 测试文件已创建并通过 CODE_PRINCIPLES 代码质量审查（~874 行，JUnit 5 + Mockito，无注释/函数超长违规）
  - **阻塞原因**: proto 配置文件存在多处 import 路径问题（`google/type/money.proto`、`web3/token.proto` 等），common 模块无法编译，测试无法运行
  - 方案: 修复 proto 依赖路径，或隔离 test scope 对 proto 模块的依赖，使测试可编译执行
