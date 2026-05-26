# TODO (2026-05-26 重新审查更新 — 二次修正)

## ❌ 未通过重新审查 — 需修复后重新勾选

### P3: 前端组件 (6个组件)
- [x] **RestockSuggestions.tsx / DemandChart.tsx / InventoryAlertPanel.tsx / ShelfHeatmap.tsx / WarehouseStatus.tsx / Warehouse3DView.tsx** ✅ 2026-05-26 修复完成
  - ✅ DemandChart.tsx - 提取所有魔法数字为 CHART_CONFIG 常量，拆分为 10+ 个子组件，每个 ≤20 行，补充 role/aria-label
  - ✅ Warehouse3DView.tsx - 提取 SHELF_CONFIG/WAREHOUSE_CONFIG/HEATMAP_CONFIG 常量，使用 styles/constants.ts 的 Colors，拆分子组件
  - ✅ InventoryAlertPanel.tsx - 移除未使用的 Badge 导入，拆分为 7 个子组件，修复三元嵌套
  - ✅ RestockSuggestions.tsx - 提取 SUGGESTION_CONFIG/URGENCY_* 常量，拆分为 5 个子组件
  - ✅ ShelfHeatmap.tsx - 提取 HEATMAP_CONFIG/LOAD_THRESHOLDS 常量，拆分为 7 个子组件
  - ✅ WarehouseStatus.tsx - compact 模式包裹 ErrorBoundary，提取 STATUS_CONFIG 常量，拆分为 9 个子组件
  - ✅ 所有组件补充完整的 role/aria-label 可访问性支持
  - ✅ 主渲染函数均已拆分至 ≤20 行

### P2: CommandService 层 create/update 方法违反"一个函数做一件事"
- [x] **WarehouseApplicationServiceImpl** ✅ 2026-05-26 部分修复
  - ✅ 提取魔法数字/字符串为常量：STATUS_ACTIVE/STATUS_PENDING, WAREHOUSE_CODE_PREFIX/INBOUND_ORDER_PREFIX, CODE_SUFFIX_LENGTH, DEFAULT_USED_CAPACITY/DEFAULT_QUANTITY
  - ✅ 移除残留注释："简化实现，返回空列表或通过其他方式实现" 和 "发布入库完成事件和入库明细事件"
  - ✅ 添加 getActualQuantity() 辅助方法，消除 lambda 内嵌套 if→for→三元
  - ⚠️ CQRS 分离（命令方法返回 void）需更大架构改动，暂未实现
  - ⚠️ Optional.orElse(null) 改为抛业务异常需更大改动，暂未实现

### 原有未完成项（保持不变）
- [ ] FallbackRouter + AlgorithmFallback 实现 (相关类不存在，暂不适用)
- [ ] DigitalTwinKafkaConsumer 单元测试 (已有集成测试 DigitalTwinKafkaConsumerTest.java)
- [ ] `digital-twin-api.test.ts` 只验证函数存在性，未测试实际 HTTP 调用 (需补充集成测试)

## 🔍 2026-05-26 复审查出新违规（二次审查新发现）

### P3: InventoryCommandService.java "INV-" 前缀硬编码
- [x] **InventoryCommandService.java** ✅ 2026-05-26 已修复
  - ✅ 添加常量 `ALERT_CODE_PREFIX = "INV-"` 并替换硬编码

### P3: DigitalTwinKafkaConsumer.java "_" 分隔符硬编码
- [x] **DigitalTwinKafkaConsumer.java** ✅ 2026-05-26 已修复
  - ✅ 添加常量 `MESSAGE_ID_DELIMITER = "_"` 并替换硬编码

### P3: LocationRecommendationFallback.java 魔法字符串残留
- [x] **LocationRecommendationFallback.java** ✅ 2026-05-26 已修复
  - ✅ 添加常量 `DEFAULT_CATEGORY = "GENERAL"` / `CLASSIFICATION_METHOD_ABC = "abc_classification"` 并替换硬编码

---

## ✅ 已通过二次审查 — 从 TODO 删除

以下项经核查确认合规，已从 TODO 中删除：
- ~~P2: supply-chain-domain WarehouseRepository save() 拆分~~ → `void insert` + `void update` 正确实现，通过规范检查 ✅
  - `WarehouseRepository.java` + `WarehouseRepositoryImpl.java` 均合规
  - `InboundOrderRepository.java` + `InboundOrderRepositoryImpl.java` 同步正确
- ~~P2: DigitalTwinKafkaConsumer InterruptedException~~ → 代码无问题，无需修复 ✅

以下为 **前次审查已通过** 的项，维持已删除状态（仅记录备查）：
- ~~项目存在原有编译错误~~ → **已修复 (2026-05-25 22:02)**
  - 根因：`AbstractAIClient.objectMapper` 为非 static 且 private，静态内部类无法访问
  - 修复：改为 `public static final ObjectMapper objectMapper`
  - 补充缺失导入：`import java.net.http.HttpRequest` (3个Client类)
  - 修复类型不兼容：`LocationRecommendationClient.getProductCorrelations()` 返回类型
- P0 数据库 Schema (`schema.sql`, 5张表+索引+外键)
- 后端仓储实体单元测试 (`WarehouseCommandServiceTest` / `ShelfCommandServiceTest` / `InventoryCommandServiceTest` / `DigitalTwinEventPublisherTest`)
- Kafka 消费集成测试 (`DigitalTwinKafkaConsumerIntegrationTest.java` CountDownLatch 修复)
- Gateway WebSocket `/ws/digital-twin` 白名单配置
- 代码审查超行修复（所有超行文件已确认修复）
- P1 WarehouseCapability 枚举 (4种能力定义清晰)
- P1 FallbackRouter 三路路由 (AI → 算法 → 人工工单)
- P0 实体层魔法数字 (Shelf.java DEFAULT_LEVEL_NUMBER/DEFAULT_TOTAL_LEVELS, InventoryItem.java CRITICAL_THRESHOLD_FACTOR)
- P0 FieldAssigner (4个) 异常日志修复
- P0 ShelfCommandService.java 魔法数字修复
- P2 RestockSuggestionFallback.java 编译错误+吞异常修复
- P2 LocationRecommendationFallback.java 吞异常修复
- DigitalTwinController.java / DigitalTwinAuthHandshakeInterceptor.java logger.debug 修复
- P0 RepositoryImpl (4个) save() 方法修复 (digital-twin 模块)
- P0 InventoryCommandService.java 多项违规修复
- P0 DigitalTwinKafkaConsumer.java 方法拆分 + InterruptedException 日志 + 移除内存去重
- P1 WarehouseCapabilityInitializer.java return 类型 + URL 配置注入 + 方法拆分 + 接口提取
- P2 RestockSuggestionFallback.java 魔法数字常量提取
- P2 LocationRecommendationFallback.java 魔法数字常量提取
- P2 DigitalTwinController.java 异常日志 + 常量抽取 + 注释移除
- 前端测试文件 useAlertNotification.test.ts / DeviceChart.test.ts 注释已移除
- **P3: InventoryCommandService.java L168 ALERT_CODE_LENGTH = 16** ✅ 原魔法数字已修复
- **P3: DigitalTwinKafkaConsumer.java L290 DEFAULT_CONFIDENCE = 0.0** ✅ 原魔法数字已修复
- **P3: LocationRecommendationFallback.java ZONE_A/B/C + ADJACENT_CATEGORY** ✅ 原魔法字符串已修复

---

## 联调测试

### 2. 集成测试
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
| **`digital-twin-service`** | ✅ 是 | ✅ 是 | ✅ 已读取并鉴权 | ✅ `@RequirePermission("dt:*")` 已覆盖 |
| **ERP 各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |
| **供应链各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |

### 需要做的工作（跨域统一方案）
1. **Gateway 白名单**：WebSocket 端点 `/ws` 需要加入 `excludedPathPatterns`（如果无需鉴权）
   - **已修复** (2026-05-25): 添加 `/ws/digital-twin` 到 `excludedPathPatterns`
2. **ERP + 供应链**：待实现 - 统一规划权限资源树，避免各域各自造轮子
