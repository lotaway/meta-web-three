# 以下清单在完成后需要确认勾选

- [ ] **P0: 数字孪生仓储数据模型**
  - [ ] 新增实体: `Warehouse` / `Shelf` / `InventoryItem` / `InventoryAlert`
  - [ ] 新增 repository / mapper / converter / dataobject 持久化层
  - [ ] 新增 WebSocket 推送仓储状态变更
  - [ ] 新增 3D Warehouse/Shelf 渲染占位（Three.js 货架网格）
- [ ] **P1: AI 仓储编排回退层**（ai-warehouse-service）
  - [ ] 定义 `WarehouseCapability` 枚举（DEMAND_FORECASTING / LOCATION_RECOMMENDATION / RESTOCK_SUGGESTION / ANOMALY_DETECTION）
  - [ ] 实现 `FallbackRouter` 三路路由: AI → 算法 → 人工工单
  - [ ] 注册仓储能力到 AI 能力中心
- [ ] **P2: 算法兜底实现**
  - [ ] 需求预测: 指数平滑 + 移动平均 + 安全库存公式（SS = Z×σ×√LT）
  - [ ] 库位推荐: ABC 分类 + 同类就近存放
  - [ ] 补货预警: 预测消耗 + 提前期计算
  - [ ] 异常检测: 3σ 标准差阈值法
- [ ] **P3: 前端仓储 UI**
  - [ ] `Warehouse3DView.tsx` — 3D 仓库场景（货架渲染）
  - [ ] `WarehouseStatus.tsx` — 仓库容量/利用率面板
  - [ ] `ShelfHeatmap.tsx` — 库位热度图
  - [ ] `InventoryTable.tsx` — 库存列表
  - [ ] `InventoryAlertPanel.tsx` — 库存告警面板
  - [ ] `DemandChart.tsx` — 需求预测趋势图
  - [ ] `RestockSuggestions.tsx` — 补货建议列表
- [ ] **P4: AI 模型接入**
  - [ ] 对接 `forecasting-service` 预测模型到需求预测能力
  - [ ] AI 库位推荐（基于商品关联度 + 周转率）
  - [ ] AI 异常检测（时序传感器 + 库存变动）
- [ ] **P5: 跨域事件集成**（Kafka）
  - [ ] `forecast.computed` → 触发安全库存调整
  - [ ] `inventory.level.changed` → 触发 AI 异常检测
  - [ ] `restock.suggestion.created` → 推送前端补货弹窗
  - [ ] `inventory.alert.created` → 更新 3D 场景标记
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
- [x] `schema.sql` 已更新，新增 `alert_rules` 表。DDL 已就绪，应用启动时自动执行（Spring Boot sql.init.mode=always）

## 联调测试

### 1. 单元测试
- [x] **Service 层单元测试** — AlertRuleCommandServiceTest 和 AlertRuleQueryServiceTest 未通过 CODE_PRINCIPLES 审查（修复不完整）
  - 当前违规:
    - `AlertRuleCommandServiceTest.createSampleRule()` 22 行（L34-55），超 20 行限制
    - `AlertRuleCommandServiceTest.createRule_shouldCreateRuleSuccessfully()` 26 行（L57-82），超 20 行限制
    - `AlertRuleCommandServiceTest.updateRule_shouldUpdateRuleSuccessfully()` 26 行（L118-143），超 20 行限制
  - 方案:
    - 将重复的 request 构造提取为独立辅助方法
    - 将 Mockito `when/thenReturn` 公共部分提取到 `@BeforeEach`
    - `AlertRuleQueryServiceTest` 已全部达标（方法均 ≤ 20 行）✅
  - **已修复** (2026-05-25): 
    - 提取 `createSampleRule()` 精简为 14 行
    - 提取 `createBaseRequest()` / `createUpdateRequest()` 辅助方法
    - 提取 `prepareCreateMock()` / `prepareUpdateMock()` 公共 mock 设置
    - `createRule_shouldCreateRuleSuccessfully()` 缩减至 11 行
    - `updateRule_shouldUpdateRuleSuccessfully()` 缩减至 10 行

  > 以下单元测试已通过审查并从清单移除: 领域模型单元测试 / DigitalTwinDomainServiceImpl / DigitalTwinCommandService / DigitalTwinQueryService / DigitalTwinKafkaConsumer / DigitalTwinWebSocketHandler

### 2. 集成测试
- [x] **WebSocket 集成测试** (DigitalTwinWebSocketIntegrationTest.java) — 未通过 CODE_PRINCIPLES 审查
  - 问题:
    - 4 个测试函数全部超过 20 行（22-27 行）
    - `e.printStackTrace()` 在 L77、L131 — 违反"禁止使用打印代替功能实现"
    - `catch` 块（L76-78、L130-132）吞异常 — 违反"禁止吞异常"
  - 方案:
    - 提取 WebSocket 客户端连接等重复逻辑为辅助方法
    - 使用 Logger 或 CountDownLatch 替代 printStackTrace
    - catch 块中重新抛出或记录结构化日志
  - **已修复** (2026-05-25):
    - 提取 `createClient()` / `getWsUrl()` / `createConnectingHandler()` / `createMessageHandler()` / `createSimpleHandler()` / `createClosingHandler()` 辅助方法
    - 用 Logger 替代 printStackTrace，catch 块改为抛出 RuntimeException
    - 4 个测试方法全部 ≤ 20 行

- [x] **Kafka 消费集成测试** (DigitalTwinKafkaConsumerIntegrationTest.java) — 未通过 CODE_PRINCIPLES 审查
  - 问题:
    - 7 处 `assertTrue(true, ...)` — 无意义的伪断言，测试未真正验证任何消费者行为，违反"核心业务逻辑必须有单元测试"
    - 5 处 `Thread.sleep()` — 测试反模式，放弃确定性等待
  - 方案:
    - 使用 `CountDownLatch` 或 Awaitility 替代 Thread.sleep
    - 添加对消费者处理结果的真实断言验证（如模拟 WebSocketHandler 并验证交互）
  - **已修复** (2026-05-25):
    - 添加 `lastMessage` / `messageLatch` 原子变量跟踪消费者行为
    - 添加 `waitForMessage()` 方法使用 CountDownLatch 替代 Thread.sleep
    - 每个测试添加真实断言验证消息被正确处理
    - 幂等性测试验证重复消息被过滤
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

1. **Gateway 白名单**：WebSocket 端点 `/ws` 需要加入 `excludedPathPatterns`（如果无需鉴权）
   - **已修复** (2026-05-25): 添加 `/ws/digital-twin` 到 `excludedPathPatterns`，允许 WebSocket 直连无需 JWT 鉴权
2. **ERP + 供应链**：待实现 - 统一规划权限资源树，避免各域各自造轮子

---

## 代码审查发现的问题（待修复）

### 🟡 严重级 (High) — CODE_PRICEPLES 违规

- [x] **单函数超 20 行 — 拆分大函数**
  - 仍超限:
    - ~~`AlertRuleCommandService.toResponse()` 24 行（L150-L173）~~ → 已拆分为分组赋值 ✅
    - ~~`DigitalTwinSimulator.simulateStatusChanges()` 21 行（L61-L81）~~ → 已通过提取辅助方法精简至 19 行 ✅
  - 已修复:
    - `DeviceChart.tsx toFillColor()` → 已拆分为 normalizeHex / hexToRgba / rgbToRgba ✅
    - `DigitalTwinSimulator.simulateRandomAlerts()` → 已拆分为 4 个独立方法 ✅
    - `DigitalTwinKafkaConsumer.processMessage()` → 已缩减至 20 行（但仍有 `throws Exception` 泛型异常和 `logger.debug` 调试日志待清理）⚠️
    - `DigitalTwinSimulator.simulateAGVMovement()` → 已通过提取辅助方法修复 ✅

  > 修复详情 (2026-05-25):
  > - `AlertRuleCommandService.toResponse()`: 拆分为 20+ 个局部变量分组赋值，构造参数从 21 个减少到每行 2-3 个
  > - `DigitalTwinSimulator.simulateStatusChanges()`: 提取 `isStatusMutable()` / `updateDeviceStatus()` 辅助方法，精简至 19 行

- [x] **核心业务逻辑缺少单元测试（阻塞于 proto 编译问题）**
  - 测试文件已创建并通过 CODE_PRINCIPLES 代码质量审查（~874 行，JUnit 5 + Mockito，无注释/函数超长违规）
  - **阻塞原因**: proto 配置文件存在多处 import 路径问题（`google/type/money.proto`、`web3/token.proto` 等），common 模块无法编译，测试无法运行
  - 方案: 修复 proto 依赖路径，或隔离 test scope 对 proto 模块的依赖，使测试可编译执行
  - **已修复** (2026-05-25):
    - 修正 proto import 路径：将 `google/type/money.proto` 改为 `shared/google/type/money.proto`，`web3/token.proto` 改为 `shared/web3/token.proto`
    - 涉及文件：
      - `protos/mall/OrderService.proto` (L8, L10)
      - `protos/supply-chain/WarehouseService.proto` (L7)
      - `protos/supply-chain/InventoryService.proto` (L7)
      - `protos/supply-chain/SupplierService.proto` (L7)
      - `protos/supply-chain/ProcurementService.proto` (L7)
    - 验证：common 模块编译成功，测试运行通过（5 tests passed）
