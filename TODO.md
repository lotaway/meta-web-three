# TODO (2026-05-26 重新审查更新 — 二次修正)

## ❌ 未通过重新审查 — 需修复后重新勾选

### P3: 前端组件 (6个组件)
- [ ] **RestockSuggestions.tsx / DemandChart.tsx / InventoryAlertPanel.tsx / ShelfHeatmap.tsx / WarehouseStatus.tsx / Warehouse3DView.tsx** ❌ 2026-05-26 二次审查未通过
  - ✅ 原有修复（移除注释、使用样式常量、ErrorBoundary、基础 UI 组件、可访问性基本支持）已确认有效
  - ❌ **DemandChart.tsx** 大量魔法数字残留（20+处，如坐标值、尺寸、留白常量未用），主渲染函数 62 行
  - ❌ **Warehouse3DView.tsx** 魔法数字泛滥、自定 COLORS 对象未使用 styles/constants.ts、函数超长（ShelfModel 34 行、HeatmapOverlay 28 行）、未用基础 UI 组件
  - ❌ **InventoryAlertPanel.tsx** Badge 死导入未使用、函数超长、连续 4 次三元嵌套违反"禁止连续三次以上 if/else"
  - ❌ **RestockSuggestions.tsx** 主函数 57 行、三元链嵌套违反规范、魔法数字（8px/500px/40px 等）
  - ❌ **ShelfHeatmap.tsx** 主函数 59 行、魔法数字（阈值/尺寸/网格）
  - ❌ **WarehouseStatus.tsx** compact 模式未包裹 ErrorBoundary、魔法数字（10px/60px 等）
  - ❌ **所有组件** 可访问性不完整：容器级 role/aria-label 普遍缺失
  - ❌ **所有组件** 主渲染函数普遍超过 20 行（6 中 5 个违规）
  - ❌ **Shadows 常量已定义但从未使用**（styles/constants.ts 死代码）
  - ❌ **utils/format.ts 的 formatPercent/clamp 从未被调用**
  - 建议：主渲染函数按子区域拆分为 ≤20 行子组件；所有 inline 尺寸值统一替换为 Spacing/Size 常量；每个容器补充 role/aria-label；Warehouse3DView 使用 styles/constants.ts 的 Colors；删除未使用的 Badge 导入和 Shadows 死代码

### P2: CommandService 层 create/update 方法违反"一个函数做一件事"
- [ ] **WarehouseApplicationServiceImpl** — 所有 `create*` / `update*` / `confirm*` / `complete*` 方法均同时返回实体对象并产生 DB 写副作用 ❌ 2026-05-26 二次审查未通过
  - ❌ `createWarehouse`(35-56, 22行): 实体赋值 + insert + 事件发布 + 查询返回，4 件事
  - ❌ `updateWarehouse`(58-78, 21行): Optional 解析 + 字段更新 + update + 查询返回
  - ❌ `createInboundOrder`(94-128, 35行): DTO→实体 + Item 列表构建 + insert + 事件发布 + 查询返回
  - ❌ `confirmInboundOrder`(131-142): confirm 逻辑 + update + 查询返回
  - ❌ `completeInboundOrder`(143-170, 28行): complete + 设置到达时间 + update + 2 个事件发布 + 查询返回
  - ❌ 所有方法同时存在**返回值 + 副作用**，违反"一个函数要么返回要么副作用"
  - ❌ 残留注释 `// 简化实现，返回空列表或通过其他方式实现`(L89) 和 `// 发布入库完成事件和入库明细事件`(L153)
  - ❌ 魔法数字/字符串：`0`(L46/L157)、`"ACTIVE"`(L47)、`"PENDING"`(L100/L117)、`"WH"`/`"IB"`/`4`(L195/L199)
  - ❌ `completeInboundOrder`(L153-164) lambda 内嵌套 if→for→三元，违反防护语句原则
  - ❌ 多处 `Optional.orElse(null)`(L77/84/139/169/176) 丢失 Optional 语义，增大 NPE 风险
  - 建议：命令方法返回 void，查询方法无副作用（CQRS 分离）；将每个方法拆分为多个 ≤20 行子方法；使用枚举替代状态字符串；Optional 应抛业务异常而非返回 null

### 原有未完成项（保持不变）
- [ ] FallbackRouter + AlgorithmFallback 实现 (相关类不存在，暂不适用)
- [ ] DigitalTwinKafkaConsumer 单元测试 (已有集成测试 DigitalTwinKafkaConsumerTest.java)
- [ ] `digital-twin-api.test.ts` 只验证函数存在性，未测试实际 HTTP 调用 (需补充集成测试)

## 🔍 2026-05-26 复审查出新违规（二次审查新发现）

### P3: InventoryCommandService.java "INV-" 前缀硬编码
- [ ] **InventoryCommandService.java** — L169 `substring(0, ALERT_CODE_LENGTH)` 中 `"INV-"` 告警码前缀为硬编码魔法字符串
  - 建议：提取为常量 `private static final String ALERT_CODE_PREFIX = "INV-"`

### P3: DigitalTwinKafkaConsumer.java "_" 分隔符硬编码
- [ ] **DigitalTwinKafkaConsumer.java** — L315 `"_"` 用于拼接 deviceCode 和 timestamp，为硬编码魔法字符串
  - 建议：提取为常量 `private static final String MESSAGE_ID_DELIMITER = "_"`

### P3: LocationRecommendationFallback.java 魔法字符串残留
- [ ] **LocationRecommendationFallback.java** — L58 `"GENERAL"` 默认分类、L88 `"abc_classification"` 分类方法名为硬编码魔法字符串
  - 建议：提取为常量 `DEFAULT_CATEGORY` / `CLASSIFICATION_METHOD_ABC`

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
