# 以下清单在完成后需要确认勾选

- [x] **P0: 数字孪生仓储数据模型** — 功能已实现但代码质量未通过 CODE_PRINCIPLES 审查
  - [x] 新增实体: `Warehouse` / `Shelf` / `InventoryItem` / `InventoryAlert`
  - [x] 新增 repository / mapper / converter / dataobject 持久化层
  - [x] 新增 WebSocket 推送仓储状态变更
  - [x] 新增 3D Warehouse/Shelf 渲染占位（Three.js 货架网格）
    - 已创建 7 个前端组件 (2026-05-25 19:45):
      - `Warehouse3DView.tsx` — 3D 仓库场景（货架网格 + 热度图）
      - `WarehouseStatus.tsx` — 仓库容量/利用率面板
      - `ShelfHeatmap.tsx` — 库位热度图
      - `InventoryTable.tsx` — 库存列表（分页+搜索+排序）
      - `InventoryAlertPanel.tsx` — 库存告警面板
      - `DemandChart.tsx` — 需求预测趋势图
      - `RestockSuggestions.tsx` — 补货建议列表
- [x] **P1: AI 仓储编排回退层**（ai-warehouse-service）
  - [x] 定义 `WarehouseCapability` 枚举（DEMAND_FORECASTING / LOCATION_RECOMMENDATION / RESTOCK_SUGGESTION / ANOMALY_DETECTION）
  - [x] 实现 `FallbackRouter` 三路路由: AI → 算法 → 人工工单
  - [x] 注册仓储能力到 AI 能力中心
- [x] **P2: 算法兜底实现**
  - [x] 需求预测: 指数平滑 + 移动平均 + 安全库存公式（SS = Z×σ×√LT）
  - [x] 库位推荐: ABC 分类 + 同类就近存放
  - [x] 补货预警: 预测消耗 + 提前期计算
  - [x] 异常检测: 3σ 标准差阈值法
- [ ] **P3: 前端仓储 UI**
  - [x] `Warehouse3DView.tsx` — 3D 仓库场景（货架渲染） ✅
  - [x] `WarehouseStatus.tsx` — 仓库容量/利用率面板 ✅
  - [x] `ShelfHeatmap.tsx` — 库位热度图 ✅
  - [x] `InventoryTable.tsx` — 库存列表 ✅
  - [x] `InventoryAlertPanel.tsx` — 库存告警面板 ✅
  - [x] `DemandChart.tsx` — 需求预测趋势图 ✅
  - [x] `RestockSuggestions.tsx` — 补货建议列表 ✅
- [ ] **P4: AI 模型接入**
  - [ ] 对接 `forecasting-service` 预测模型到需求预测能力
  - [ ] AI 库位推荐（基于商品关联度 + 周转率）
  - [ ] AI 异常检测（时序传感器 + 库存变动）
- [ ] **P5: 跨域事件集成**（Kafka）
  - [ ] `forecast.computed` → 触发安全库存调整
  - [ ] `inventory.level.changed` → 触发 AI 异常检测
  - [ ] `restock.suggestion.created` → 推送前端补货弹窗
  - [ ] `inventory.alert.created` → 更新 3D 场景标记
- [x] 是否完善了所必须的自动化测试 — 前端已覆盖但后端仓储核心业务逻辑零测试
  - **已添加的前端测试** (2026-05-25, 27 tests passed ✅):
    - 安装 vitest、@testing-library/react、@testing-library/jest-dom、jsdom 测试基础设施
    - `useAlertNotification.test.ts` - 5 tests (hook测试) ✅
    - `digital-twin-api.test.ts` - 8 tests (覆盖不足: 仅验证函数存在性，未测试实际HTTP调用) ⚠️
    - `DeviceChart.test.ts` - 14 tests (纯函数测试全面) ✅
  - **后端仓储模块单元测试** (2026-05-25 19:45) ✅:
    - `WarehouseTest.java` - 10 tests (状态机 + 利用率计算) ✅
    - `ShelfTest.java` - 13 tests (重量管理 + 状态转换) ✅
    - `InventoryItemTest.java` - 16 tests (库存状态 + 补货计算) ✅
    - `InventoryAlertTest.java` - 13 tests (生命周期 + 告警升级) ✅

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

## P0 任务完成进度 (2026-05-25 18:35)

### 已完成 ✅
1. **实体层** (domain/entity):
   - `Warehouse.java` - 仓库实体，含状态机、面积利用率计算
   - `Shelf.java` - 货架实体，含状态管理、重量计算
   - `InventoryItem.java` - 库存物料实体，含状态更新、保质期检查
   - `InventoryAlert.java` - 库存告警实体，含生命周期管理

2. **Repository 接口** (domain/repository):
   - `WarehouseRepository.java`
   - `ShelfRepository.java`
   - `InventoryItemRepository.java`
   - `InventoryAlertRepository.java`

3. **数据对象** (infrastructure/persistence/dataobject):
   - `WarehouseDO.java`
   - `ShelfDO.java`
   - `InventoryItemDO.java`
   - `InventoryAlertDO.java`

4. **Mapper 接口** (infrastructure/persistence/mapper):
   - `WarehouseMapper.java`
   - `ShelfMapper.java`
   - `InventoryItemMapper.java`
   - `InventoryAlertMapper.java`

5. **转换器** (infrastructure/persistence/converter):
   - `WarehouseConverter.java`
   - `ShelfConverter.java`
   - `InventoryItemConverter.java`
   - `InventoryAlertConverter.java`

6. **Repository 实现** (infrastructure/persistence/repository):
   - `WarehouseRepositoryImpl.java`
   - `ShelfRepositoryImpl.java`
   - `InventoryItemRepositoryImpl.java`
   - `InventoryAlertRepositoryImpl.java`

7. **应用层命令服务** (application/command):
   - `WarehouseCommandService.java` - 仓库创建/更新/激活/退役
   - `ShelfCommandService.java` - 货架创建/占用/清空
   - `InventoryCommandService.java` - 库存管理、低库存告警自动创建

8. **应用层查询服务** (application/query):
   - `WarehouseQueryService.java` - 仓库查询、利用率计算
   - `ShelfQueryService.java` - 货架查询
   - `InventoryQueryService.java` - 库存和告警查询

9. **数据库 Schema** (schema.sql):
   - 新增 `warehouses` 表
   - 新增 `shelves` 表
   - 新增 `inventory_items` 表
   - 新增 `inventory_alerts` 表
   - 新增 `inventory_movement_logs` 表
   - 新增相关索引和外键约束

10. **WebSocket 推送仓储状态变更** (2026-05-25 18:35) ✅
    - `DigitalTwinEventPublisher.java` 新增方法:
      - `publishWarehouseStatusChanged()` - 仓库状态变更事件
      - `publishInventoryLevelChanged()` - 库存水位变更事件
      - `publishInventoryAlertCreated()` - 库存告警创建事件
      - `publishRestockSuggestionCreated()` - 补货建议创建事件
      - `publishShelfStatusChanged()` - 货架状态变更事件
    - `DigitalTwinKafkaConsumer.java` 新增 Kafka 监听器:
      - `consumeWarehouseStatusChanged()` - 监听 warehouse.status.changed
      - `consumeInventoryLevelChanged()` - 监听 inventory.level.changed
      - `consumeInventoryAlertCreated()` - 监听 inventory.alert.created
      - `consumeRestockSuggestionCreated()` - 监听 restock.suggestion.created
      - `consumeShelfStatusChanged()` - 监听 shelf.status.changed
    - 命令服务集成事件发布:
      - `WarehouseCommandService` - activate/decommission 时发布事件
      - `InventoryCommandService` - addStock/removeStock/告警创建时发布事件
      - `ShelfCommandService` - occupy/clear 时发布事件
    - Kafka 消费者处理后通过 WebSocket 广播到前端

### 待完成 ⚠️
- 3D Warehouse/Shelf 渲染占位（Three.js 货架网格）

### P1 任务完成进度 (2026-05-25 18:55)

#### 已完成 ✅

1. **WarehouseCapability 枚举** (domain/entity):
   - `WarehouseCapability.java` - 定义四种仓储能力:
     - DEMAND_FORECASTING (需求预测) - 默认 ALGORITHM 兜底
     - LOCATION_RECOMMENDATION (库位推荐) - 默认 ALGORITHM 兜底
     - RESTOCK_SUGGESTION (补货建议) - 默认 HUMAN 兜底
     - ANOMALY_DETECTION (异常检测) - 默认 ALGORITHM 兜底

2. **FallbackRouter 三路路由** (infrastructure/router):
   - `FallbackRouter.java` - 实现 AI → 算法 → 人工工单三级路由:
     - tryAI(): 优先调用 AI 服务，成功则返回
     - tryAlgorithm(): AI 失败时调用 AlgorithmFallback 实现
     - tryHuman(): 算法失败时创建人工工单
   - RouteResult 封装路由结果和使用的路由类型
   - AlgorithmFallback 接口规范算法兜底实现

3. **算法兜底实现** (infrastructure/algorithm):
   - `DemandForecastingFallback.java` - 需求预测:
     - 指数平滑 (α=0.3)
     - 移动平均 (窗口=3)
     - 安全库存公式 (SS = Z×σ×√LT, Z=1.65)
   - `LocationRecommendationFallback.java` - 库位推荐:
     - ABC 分类 (A区: velocity≥80, B区: velocity≥50, C区: velocity<50)
     - 同类就近存放
   - `RestockSuggestionFallback.java` - 补货预警:
     - 预测消耗计算
     - 提前期计算 (默认7天)
     - 紧急程度判断
   - `AnomalyDetectionFallback.java` - 异常检测:
     - 3σ 标准差阈值法
     - SPIKE/DROP/NORMAL 类型判断

4. **能力中心注册** (infrastructure/config):
   - `WarehouseCapabilityInitializer.java` - 启动时自动注册:
     - @EventListener(ApplicationReadyEvent) 触发注册
     - 幂等检查避免重复注册
     - 读取 fallbackConfig 构建配置

### 阻塞问题 ⚠️
- 项目存在原有编译错误（AlertRule 引用不存在的 AlertType、PermissionChecker、ErrorCode、RequirePermission 类），需先修复

---

## 联调测试

### 1. 单元测试

### 2. 集成测试

- [ ] **Kafka 消费集成测试** (DigitalTwinKafkaConsumerIntegrationTest.java) — 修复不完整
  - 修复进展:
    - assertTrue(true, ...) 伪断言 → 已替换为真实断言 ✅
    - 大部分 Thread.sleep → 已替换为 CountDownLatch ✅
    - 幂等性测试已添加 ✅
  - **仍存在问题**:
    - `idempotencyTest` 仍使用 `Thread.sleep(1000)` → 放弃确定性等待，测试脆弱
    - 多个测试共享同一个 `CountDownLatch` 类字段 → 并行测试下竞态条件风险
  - 建议: 将 idempotencyTest 的 sleep 改为 CountDownLatch + 唯一消息ID去重验证; latch 改为方法级局部变量
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

- [x] **单函数超 20 行 — 拆分大函数** — 已全部修复 (2026-05-25 19:36)
  - **已确认修复（保留为参考）**:
    - `AlertRuleCommandService.toResponse()` — 已拆分为分组赋值 ✅
    - `DigitalTwinSimulator.simulateStatusChanges()` — 已精简至 19 行 ✅
    - `DeviceChart.tsx toFillColor()` — 已拆分 ✅
    - `DigitalTwinSimulator.simulateRandomAlerts()` — 已拆分为 4 个独立方法 ✅
    - `DigitalTwinSimulator.simulateAGVMovement()` — 已通过提取辅助方法修复 ✅
  - **已修复 Converter 超行问题**:
    - `WarehouseConverter.toEntity()` / `toDO()` — 提取 WarehouseFieldAssigner ✅
    - `ShelfConverter.toEntity()` / `toDO()` — 提取 ShelfFieldAssigner ✅
    - `InventoryItemConverter.toEntity()` / `toDO()` — 提取 InventoryItemFieldAssigner ✅
    - `InventoryAlertConverter.toEntity()` / `toDO()` — 提取 InventoryAlertFieldAssigner ✅
  - **已修复 CommandService/QueryService 超行问题**:
    - `WarehouseCommandService.updateWarehouse()` — 提取 applyUpdates() ✅
    - `ShelfCommandService.createShelf()` — 提取 validateShelfRequest() / initializeShelfFields() ✅
    - `InventoryCommandService.createItem()` — 提取 validateCreateItemRequest() / assignItemFields() ✅
    - `InventoryCommandService.updateItem()` — 提取 applyItemUpdates() ✅
    - `InventoryCommandService.checkAndCreateLowStockAlert()` — 提取 createLowStockAlert() ✅
    - `InventoryQueryService.getInventoryAlertSummaries()` — 提取 toAlertSummary() ✅
  - **已修复 KafkaConsumer 超行问题**:
    - `DigitalTwinKafkaConsumer.processMessage()` — 添加 try-catch 包裹，约 20 行 ✅

- [ ] **核心业务逻辑缺少单元测试** — 仓储核心业务逻辑零测试覆盖
  - 建议: 参照 AlertRuleCommandServiceTest/AlertRuleQueryServiceTest 的模式，为仓储模块编写单元测试