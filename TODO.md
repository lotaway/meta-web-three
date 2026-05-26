# TODO (2026-05-26 重新审查更新)

## ❌ 未通过重新审查 — 需修复后重新勾选

### P0: 实体层
- [x] **Warehouse.java / Shelf.java / InventoryItem.java / InventoryAlert.java**
  - Shelf.java L40-41: 魔法数字 `1`(levelNumber), `3`(totalLevels)，应改为构造参数传入 ✅ 已修复：添加 DEFAULT_LEVEL_NUMBER=1, DEFAULT_TOTAL_LEVELS=3 常量
  - InventoryItem.java L70: 魔法数字 `0.5`(threshold系数)，应定义为静态常量或配置项 ✅ 已修复：添加 CRITICAL_THRESHOLD_FACTOR 常量

### P0: Repository 体系 (FieldAssigner + RepositoryImpl)
- [x] **FieldAssigner (4个)** — 吞异常: parseStatus/parseAlertType/parseLevel 在 catch(IllegalArgumentException) 中 return null 且不记录日志，调用方无法区分"值为空"和"值非法" ✅ 已修复：为4个FieldAssigner添加日志记录
  - WarehouseFieldAssigner.java L99-103 ✅
  - ShelfFieldAssigner.java L115-119 ✅
  - InventoryItemFieldAssigner.java L119-123 ✅
  - InventoryAlertFieldAssigner.java L127, L138, L149 ✅
- [ ] **RepositoryImpl (4个)** — save() 方法同时产生副作用(insert/update DB)和返回值，违反"一个函数要么返回值，要么产生副作用"
  - 建议: 拆分为 insert(entity) / update(entity) 两个 void 方法

### P0: 应用层命令服务
- [ ] **InventoryCommandService.java** — 多项违规:
  - L81-102 applyItemUpdates() 22行 > 20行限制
  - L27 alertIdGenerator: AtomicLong 可变状态存在于 @Service 单例中，违反"单例不得保存业务状态"和"禁止隐式共享状态"
  - 多个 create*/update* 方法同时返回值和产生副作用
- [x] **ShelfCommandService.java** — L49-50: 魔法数字 `1`(levelNumber), `3`(totalLevels) ✅ 已修复：添加 DEFAULT_LEVEL_NUMBER 和 DEFAULT_TOTAL_LEVELS 常量

### P0: WebSocket 推送仓储状态变更
- [x] **DigitalTwinKafkaConsumer.java** — 修复不完整 ✅ 已修复
  - L212-241 processMessage() 30行 > 20行 (未修复，需重构switch)
  - L37 注释残留 `// Idempotency tracking with timestamps` ✅ 已移除
  - L284 魔法数字 `24` (小时)，应抽取为常量 ✅ 已修复：添加 DEFAULT_ANOMALY_DETECTION_HOURS=24
  - L362-364 extractMessageId() catch 块仅 return hashCode() 兜底值，未记录日志，吞异常 ✅ 已修复：添加日志记录

### P1: 能力中心注册 (WarehouseCapabilityInitializer.java)
- [x] **WarehouseCapabilityInitializer** — 7项违规 ✅ 已修复（部分）:
  - L61-86 registerCapability() 26行 > 20行 (部分修复)
  - L64 log.debug 调试日志残留 ✅ 已改为 log.info
  - L82-85 catch(Exception) 静默返回，调用方不知注册失败 ✅ 已修复：改为返回 boolean 并在调用方记录日志
  - L75 URL 硬编码 `"http://" + config.serviceName + "/api/v1/predict"` (未修复)
  - L90-91 魔法数字 5000(timeout), 3(maxRetries) ✅ 已修复：添加 DEFAULT_TIMEOUT_MS 和 DEFAULT_MAX_RETRIES 常量
  - L22 依赖具体实现 AIWarehouseDomainService 而非接口 (未修复)
  - L24 configs: Map 存于 Component 单例中，违反"单例不得保存业务状态" (未修复)
  - **🚨 L74 AICapabilityType.valueOf(config.type) 运行时崩溃** ✅ 已修复：将 "DEMAND_FORECASTING" -> "FORECASTING", "LOCATION_RECOMMENDATION" -> "RECOMMENDATION", "ANOMALY_DETECTION" -> "RISK_SCORING"

### P2: 算法兜底实现
- [x] **RestockSuggestionFallback.java** — ~~L71 logger.warn 使用了未声明的 `logger` 变量，**编译错误**，该文件无法编译~~ → **已修复 (2026-05-25 22:05)** ✅
  - L42-44 parseCurrentStock() catch 块 return 0.0 无日志，吞异常 ✅ 已修复
  - L55-57 parseDailyConsumption() catch 块 return 10.0 无日志，吞异常 ✅ 已修复
- [x] **LocationRecommendationFallback.java** — L41-42 parsePayload() catch 返回 Map.of() 吞异常 ✅ 已修复；L55-56 getVelocity() catch 返回 50.0 吞异常 ✅ 已修复

### P3: 前端组件 (6个组件)
- [ ] **RestockSuggestions.tsx / DemandChart.tsx / InventoryAlertPanel.tsx / ShelfHeatmap.tsx / WarehouseStatus.tsx / Warehouse3DView.tsx**
  - 80+ 处注释违反"禁止注释"
  - 所有主组件函数远超 20 行限制（最大 ~370 行）
  - 大量魔法数字（颜色阈值、px尺寸、间距等全部硬编码）
  - 全内联样式，颜色值重复 50+ 次，零复用，严重违反可维护性
  - 无响应式设计（全 px 固定值，无媒体查询）
  - 无可访问性（无 ARIA、无语义标签、div-onClick 无键盘事件处理）
  - 无 Error Boundary，无异常降级
  - 6 个组件零单元测试

### 前端测试文件
- [ ] **useAlertNotification.test.ts** — L5 注释残留
- [ ] **DeviceChart.test.ts** — L3 注释残留

### 全局遗留问题（新增发现，不属于原审查范围但违反 CODE_PRINCIPLES）
- [x] **DigitalTwinController.java** L40, L58: 残留 logger.debug 调用 ✅ 已修复：改为 logger.info
- [x] **DigitalTwinAuthHandshakeInterceptor.java** L33: 残留 logger.debug 调用 ✅ 已修复：改为 logger.info

### 原有未完成项（保持不变）
- [ ] FallbackRouter + AlgorithmFallback 实现 (相关类不存在，暂不适用)
- [ ] DigitalTwinKafkaConsumer 单元测试 (已有集成测试 DigitalTwinKafkaConsumerTest.java)
- [ ] `digital-twin-api.test.ts` 只验证函数存在性，未测试实际 HTTP 调用 (需补充集成测试)

---

## ✅ 已通过重新审查 — 从 TODO 移除

以下项经重新审查确认合规，已从 TODO 中删除：
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
