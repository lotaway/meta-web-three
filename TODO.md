# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### [Backend Admin Missing]

The following backend services have been created, but lack corresponding admin and operation pages. Each needs to be added:

- mall-domain (11 services, most missing admin pages)
- ai-domain (3 services: ai-warehouse, forecasting, risk-scorer)
- factory-domain / mes-service (production management admin)
- blockchain-domain (2 services)
- erp-domain (6 services: finance, HR, invoice, project, report, settlement)
- platform-domain (7 services: commission, customer service, data analysis, media, message, notification, user behavior)
- supply-chain-domain (6 services: inventory alert, inventory, logistics, procurement, supplier, warehouse)

#### 供应链领域 (中优先级)

- [ ] **[Supply Chain] 退货管理 (RMA - Return Material Authorization)**: 逆向物流模块
   - [x] **函数过长已修复**：`createRma()`(18行)/`recordInspection()`(16行) 已通过，`toOrderDTO()` 已拆分为 5 个辅助方法（主方法 8 行）
   - [ ] **仍存在问题**：
     - [ ] **零单元测试**：核心退换货流程无任何测试
     - [ ] **Proto/Java DTO 字段不匹配**：`RecordInspectionRequest` 缺 `totalInspected`/`totalPassed`/`totalFailed`；`MakeDispositionRequest` 缺 `dispositionBy`/`scrapQuantity`/`scrapReason`；多个 DTO 缺 `createdAt`/`updatedAt` 时间戳字段。gRPC 入口会导致反序列化失败
     - [x] **makeDisposition 缺 rmaId**：`disposition` 对象的 `rmaId` 已设置为 `rmaId` 再传给领域服务
     - [x] **事件类型字符串硬编码**：已提取为 `EVENT_RMA_CREATED` 等事件常量枚举
     - [x] **getRma()/getRmaByNo() 返回 null**：已改用 `orElseThrow()` 抛明确异常
     - [x] **前端 formatAmount 零值 bug**：`if (!amount)` 已改为 `if (amount === null || amount === undefined)`，零值正常显示 `$0.00`
     - [x] **前端 conclusion 应为 select**：已改用 `<el-select>` 配合 `INSPECTION_CONCLUSION` 枚举值（NO_ISSUE/MINOR_DEFECT/MAJOR_DEFECT）
     - [x] **前端 i18n 不完整**：RmaInspectionDialog/RmaDispositionDialog 硬编码标签已替换为 i18n 翻译键

- [ ] **[Supply Chain] 分布式订单管理 (DOM)**: 跨仓库订单承诺、寻源和履行
   - [x] **`createDomOrder()` 已拆分**：`buildOrderEntity()`(15行)/`buildOrderLines()`(15行)/`processAvailabilityAndSourcing()`(16行) 已通过，`sourceOrder()` 已拆分为 `maybeCreateFulfillmentPlan()` 辅助方法（主方法 9 行），`toDomOrderDTO()` 已拆分为 `mapDomOrderFields()` 辅助方法（主方法 8 行）
   - [x] **死代码已清理**：`infrastructure/rpc/` 下 `WarehouseServiceRpcClient.java` 和 `InventoryServiceRpcClient.java` 死文件已删除
   - [ ] **仍存在问题**：
     - [ ] **零单元测试**：核心 ATP 检查、寻源算法无测试
     - [ ] **生产级模拟客户端不可接受**：`WarehouseServiceMockClient` 和 `InventoryServiceMockClient` 带有 `@Primary` 注解，是激活的模拟实现，含硬编码数据和 `Math.random()` 非确定性逻辑，必须替换为真实 RPC 客户端
     - [ ] **DomSourcingProperties 硬编码仓库 ID**：`warehouseIds = Arrays.asList(1L, 2L, 3L)` 将特定生产数据嵌入代码默认值，配置未覆盖时静默使用错误仓库列表
     - [ ] **DomSequenceGeneratorImpl 重启重复风险**：`AtomicLong` 从 0 开始，JVM 重启后同一天可能生成重复订单号

#### ERP 领域 (中优先级)

- [ ] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [x] **DDD 分层违规已修复**：`ContactController`/`CampaignController` 已改为注入应用服务，`CrmSubgraphConfig.java` 已改为注入 `CampaignQueryService`/`ContactQueryService` 替代 Repository
   - [ ] **仍存在问题**：
     - [ ] **文件超 500 行**：`CrmSubgraphConfig.java` 550 行（含 5 个内部 DTO 类）
     - [ ] **领域层依赖 MyBatis Plus**：所有 Repository `extends BaseMapper<T>` 引入 ORM 到领域层
     - [ ] **应用层依赖 MyBatis Plus**：`LeadQueryService`/`OpportunityQueryService`/`TicketQueryService` 导入 `LambdaQueryWrapper`
     - [ ] **零单元测试**：无 `src/test/` 目录
     - [x] **CrmSubgraphController.execute() 已拆分**：已提取 `buildExecutionInput()`/`buildResponse()` 辅助方法，主方法降至 6 行

- [ ] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [x] **函数超 20 行已修复**：`appendSalesReportContent` 已拆分通过，`appendWorkingCapitalSummary()` 已拆分为 `computeWorkingCapital()`/`computeCurrentRatio()`，`sendDingTalk()` 已拆分为 `buildDingTalkRequest()`，`index.vue` 的 `loadSalesData()` 已拆分为 `loadSalesTrendRow()`/`loadCategoryData()`/`loadRegionData()`，`sales.vue` 的 `loadData()` 已拆分为 `loadSalesTrend()`/`loadCategoryData()`/`loadRegionalData()`
   - [x] **缺少输入校验已修复**：`OlapController` 所有 7 个 GET 端点（`drillDown`/`rollUp`/`slice`/`pivot`/`getSalesFunnel`/`getCohortRetention`/`getTopN`）已添加 `@NotNull`/`@NotEmpty` 校验
   - [ ] **仍存在问题**：
     - [ ] **sendEmail() 为伪代码**：仅记录日志但未真正调用邮件发送 API（无 JavaMailSender），emailEnabled=false 时静默跳过，应接入真实邮件发送实现
     - [x] **前端 * 0.6 硬编码成本估算**：`index.vue` 和 `financial.vue` 中 `totalCost` 已改为从后端响应字段 `totalCost`/`todayCost` 读取
     - [ ] **DDD 分层违规**：`application/query/` 服务包含领域逻辑（毛利率计算、报表聚合）
     - [ ] **领域层使用 Lombok**：`@Builder`/`@Getter`/`@ToString` 在领域实体中
     - [ ] **零单元测试**：无测试覆盖
     - [ ] **前端缺少库存/生产专用页面**：TODO 要求 views/bi/* 系列页面，但库存和生产嵌入在 index.vue 标签页
     - [x] **测试 ErpMesIntegrationTest.testMesTaskCompletedEventIsLogged 已添加断言**：使用 awaitility 轮询验证成本记录已创建
     - [x] **测试已改用 awaitility**：`Thread.sleep()` 已全部替换为 `Awaitility.await()` 轮询

- [ ] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [ ] **仍存在问题**：
     - [ ] **通过修改现有代码实现**：`WorkOrderCommandService` 被大幅重构，添加了非原有职责的方法
     - [ ] **零单元测试**：`MesDomainServiceImpl`、`ProductionEventProcessor`、`MesCrossDomainEventPublisher` 均无测试
     - [x] **testMesTaskCompletedEventIsLogged 已添加断言**：使用 awaitility 轮询验证成本记录创建
     - [x] **测试已改用 awaitility**：`Thread.sleep()` 已全部替换为 `Awaitility.await()` 轮询

### [GitHub Issues]

- [ ] **[#1] Solana 商城模板集成**: Token, NFT, SFT 创建与管理；Token 作为商品销售；活动和佣金功能
  - [ ] Token/NFT/SFT 创建和管理合约
  - [ ] Token 作为商品销售的商城前端集成
  - [ ] 活动与佣金功能
  - 链接: https://github.com/lotaway/meta-web-three/issues/1

# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

