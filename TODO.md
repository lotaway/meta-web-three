# TODO (2026-05-26 重新审查更新)

## ❌ 未通过重新审查 — 需修复后重新勾选

### P0: Repository 体系 (FieldAssigner + RepositoryImpl)
- [x] **RepositoryImpl (4个)** — save() 方法同时产生副作用(insert/update DB)和返回值，违反"一个函数要么返回值，要么产生副作用"
  - ✅ 已修复：拆分为 insert(entity) / update(entity) 两个 void 方法
  - 修改文件：WarehouseRepository.java, ShelfRepository.java, InventoryItemRepository.java, InventoryAlertRepository.java (接口)
  - 修改文件：WarehouseRepositoryImpl.java, ShelfRepositoryImpl.java, InventoryItemRepositoryImpl.java, InventoryAlertRepositoryImpl.java (实现)
  - 修改文件：WarehouseCommandService.java, ShelfCommandService.java, InventoryCommandService.java (调用方)

### P0: 应用层命令服务
- [x] **InventoryCommandService.java** — 多项违规:
  - ✅ L81-102 applyItemUpdates() 16行 < 20行限制（已优化）
  - ✅ L27 alertIdGenerator: AtomicLong 已改为 UUID，无可变状态
  - ✅ 多个 create*/update* 方法返回值和产生副作用已修复（随 RepositoryImpl 一起修复）
### P0: WebSocket 推送仓储状态变更
- [ ] **DigitalTwinKafkaConsumer.java** — 部分修复，仍有问题:
  - L212-241 processMessage() 30行 > 20行 (未修复，需重构switch)
  - L277-302 triggerAnomalyDetection() 26行 > 20行 ❌ (新发现)
  - L68-71 InterruptedException 捕获后未记录日志，吞异常 ❌ (新发现)
  - L39 processedMessageIds: ConcurrentHashMap 存于 @Component 单例，违反"单例不得保存业务状态" ❌ (新发现)
  - L37 注释残留 ✅ 已移除；L284 魔法数字 ✅ 已修复；L362-364 异常日志 ✅ 已修复
  - 建议: processMessage/triggerAnomalyDetection 拆分子方法，InterruptedException 添加日志，幂等性改为外部去重服务

### P1: 能力中心注册 (WarehouseCapabilityInitializer.java)
- [ ] **WarehouseCapabilityInitializer** — 部分修复，仍有严重问题:
  - L72 `return;` 在 boolean 方法中 → **编译错误** ❌❌ (新发现)
  - L81 `"http://" + config.serviceName + "/api/v1/predict"` 硬编码 URL (未修复，且使用不安全的HTTP)
  - L67-94 registerCapability() 28行 > 20行 (未修复)
  - L22 依赖具体实现 AIWarehouseDomainService 而非接口 (未修复)
  - L24 configs: Map 存于 Component 单例中 (未修复)
  - L64 log.debug ✅ 已修复；L82-85 catch 静默返回 ✅ 已修复；L90-91 魔法数字 ✅ 已修复；L74 valueOf 崩溃 ✅ 已修复
  - 建议: L72 改为 return true/false；URL 改为配置注入；registerCapability 拆分子方法；AIWarehouseDomainService 提取接口

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

### P2: 算法兜底实现（新发现魔法数字）
- [ ] **RestockSuggestionFallback.java** — 新增魔法数字:
  - L54/L59/L62: 默认日消耗量 `10.0`，应定义为常量
  - L88: 最大库存乘数 `2`，应定义为常量
  - L95: 紧急阈值 `0.5`，应定义为常量
- [ ] **LocationRecommendationFallback.java** — 新增魔法数字:
  - L69: 高周转阈值 `80`，应定义为常量
  - L71: 中周转阈值 `50`，应定义为常量

### P2: DigitalTwinController.java（新发现违规）
- [ ] **DigitalTwinController.java** — 新发现违规:
  - L62-63: IllegalArgumentException 捕获后未记录日志，吞异常
  - L86-87: 魔法数字 `0.0`(默认Z坐标/旋转值)，应定义为常量
  - L33/L122/L143/L165: 注释残留 (`// Device endpoints` 等)
  - 建议: 添加异常日志、抽取常量、移除注释

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
- P0 实体层魔法数字 (Shelf.java DEFAULT_LEVEL_NUMBER/DEFAULT_TOTAL_LEVELS, InventoryItem.java CRITICAL_THRESHOLD_FACTOR) ✅ 2026-05-26 审查通过
- P0 FieldAssigner (4个) 异常日志修复 (WarehouseFieldAssigner/ShelfFieldAssigner/InventoryItemFieldAssigner/InventoryAlertFieldAssigner) ✅ 2026-05-26 审查通过
- P0 ShelfCommandService.java 魔法数字修复 (DEFAULT_LEVEL_NUMBER/DEFAULT_TOTAL_LEVELS) ✅ 2026-05-26 审查通过
- P2 RestockSuggestionFallback.java 编译错误+吞异常修复 ✅ 2026-05-26 审查通过（新发现魔法数字已另立条目）
- P2 LocationRecommendationFallback.java 吞异常修复 ✅ 2026-05-26 审查通过（新发现魔法数字已另立条目）
- DigitalTwinController.java / DigitalTwinAuthHandshakeInterceptor.java logger.debug 修复 ✅ 2026-05-26 审查通过（DigitalTwinController 新发现违规已另立条目）

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
