# TODO

[Guideline](./README.md)

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

- [x] **[Supply Chain] 退货管理 (RMA - Return Material Authorization)**: 逆向物流模块
   - [x] 定义领域实体：`RmaOrder`, `RmaOrderItem`, `RmaInspection`, `RmaDisposition`, `ReturnShipping`
   - [x] 实现退换货流程：退货申请 → 质检 → 处理决定(退款/换货/维修) → 返仓/报废
   - [x] 实现与库存服务集成：RMA 质检通过后自动触发入库
   - [x] 实现与结算服务集成：退款自动触发结算
   - [x] 添加 REST API + Protobuf 定义
   - [x] 添加 Protobuf 定义：`protos/supply-chain/RmaService.proto`
   - [x] 添加后台管理页面：`views/rma/*` 系列页面

- [x] **[Supply Chain] 分布式订单管理 (DOM)**: 跨仓库订单承诺、寻源和履行
   - [x] 定义领域实体：`DomOrder`, `DomOrderLine`, `FulfillmentPlan`, `SourcingRule`
   - [x] 实现订单寻源：根据库存、距离、成本自动选择最优仓库
   - [x] 实现可用量承诺(ATP)检查
   - [x] 添加 REST API
   - [x] 添加后台管理页面

#### ERP 领域 (中优先级)

- [ ] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [ ] 创建 `crm-service` 模块（基于 backend-api 父 POM）
   - [ ] 定义领域实体：`Lead`, `Opportunity`, `SalesPipeline`, `CustomerServiceTicket`, `Campaign`, `Contact`
   - [ ] 实现销售管道：商机创建 → 阶段推进 → 赢单/输单
   - [ ] 实现客户服务工单：创建 → 分配 → 处理 → 关闭
   - [ ] 实现与用户服务集成：客户数据同步
   - [ ] 添加 REST API + GraphQL 端点
   - [ ] 添加 Protobuf 定义：`protos/erp/CrmService.proto`
   - [ ] 添加后台管理页面：`views/crm/*` 系列页面

- [ ] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [ ] 基于现有 `reporting-service` 扩展：添加 OLAP 聚合查询能力
   - [ ] 实现销售分析看板：销量趋势、品类分布、区域对比
   - [ ] 实现财务分析看板：收入/成本/利润趋势、预算执行率
   - [ ] 实现库存分析看板：周转率、ABC分析、安全库存预警
   - [ ] 实现生产分析看板：OEE、良品率、计划达成率
   - [ ] 添加后台 BI 看板页面：`views/bi/*` 系列页面

#### 跨领域基础能力

- [ ] **[Cross] Protobuf 定义补齐**: 当前缺少 ERP 和 MES 领域的 protobuf 定义
   - [ ] 创建 `protos/erp/` 目录：定义 FinanceService, HrmService, InvoiceService, CrmService
   - [ ] 创建 `protos/factory/` 目录：定义 MesService, SchedulingService, QualityService, EquipmentService
   - [ ] 运行 `make` 生成多语言接口代码

- [ ] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [ ] ERP 生产订单发布时发送领域事件(MQ/Kafka)到 MES
   - [ ] MES 工单报工完成后发送完工事件到 ERP
   - [ ] 财务成本核算模块监听完工事件自动归集成本
   - [ ] 实现端到端集成测试

### [Digital Twin UI 交互流程未完整恢复]（新任务）

- [ ] **[Digital Twin] 右侧面板 Tab 与告警/通知/SCADA/追溯功能链路未完整闭环**
  - [ ] 在 `apps/digital-twin/system-management/src/renderer/pages/DigitalTwinPage.tsx` 中确认并恢复完整 UI 流程：
    - [ ] ToastContainer（告警/动作反馈）渲染挂载
    - [ ] 右侧 Tab：规则配置（rules）、SCADA（scada）、追溯（trace）三块的入口与状态切换
    - [ ] rules Tab：`AlertRuleList` 规则配置列表能正常加载与分页/编辑（若存在交互）
    - [ ] scada Tab：`ScadaPanel` 能基于选中设备展示遥测与指令下发
    - [ ] trace Tab：`TraceabilityPanel` 能展示完整链路并支持正向/反向追溯
    - [ ] 设备未选中时的占位逻辑从“点击提示”改为可指导的体验（并确保不会影响三块 Tab 的挂载）
    - [ ] charts Tab 的声音/通知开关（soundEnabled/notifEnabled）与 `AudioMonitor`/通知告警触发的接入
  - [ ] 校验环境变量/端口提示：异常文案应基于 `DIGITAL_TWIN_API_BASE_URL` 推导端口，而非写死 10102。

### [扫描发现的虚假/占位符/内存实现]（新增）

基于 `server/` 全量 Java 源代码扫描，发现以下未在 TODO 中记录的虚假/占位符实现，按优先级排列。

#### Critical — 内存 Map Repository（重启丢数据，无真实数据库）

- [ ] **[Critical] AI 域 RouteOptimizer 内存 Repository**: `ai-domain/route-optimizer/.../VehicleRepositoryImpl.java` 和 `RoutePlanRepositoryImpl.java` — 使用 ConcurrentHashMap 存储，系统重启即丢失，且无 JPA/MyBatis 注解
- [ ] **[Critical] AI 域 ForecastingService 内存 Repository**: `ai-domain/forecasting-service/.../SalesHistoryRepositoryImpl.java`、`SalesForecastRepositoryImpl.java`、`ForecastModelRepositoryImpl.java` — ConcurrentHashMap + AtomicLong 自增 ID 模拟持久化，无数据库配置
- [ ] **[Critical] Blockchain WalletService 内存 Repository**: `blockchain-domain/wallet-service/.../WalletRepositoryImpl.java` — ConcurrentHashMap 存储，重启丢失所有钱包数据
- [ ] **[Critical] SupplyChain 库存盘点内存 Repository**: `supply-chain-domain/inventory-service/.../StockCheckRepositoryImpl.java` 等系列 — ConcurrentHashMap 存储，无真实盘点记录持久化
- [ ] **[Critical] Common AuditLogRepository**: `common/.../AuditLogRepository.java` — ConcurrentHashMap 存储，审计日志重启即清空

#### Critical — 伪算法/占位符实现

- [ ] **[Critical] ForecastingDomainServiceImpl 伪训练**: `ai-domain/forecasting-service/.../ForecastingDomainServiceImpl.java` — 使用 `Math.random()` 模拟训练精度和预测结果，无真实 ML 模型调用
- [ ] **[Critical] RouteOptimizerDomainServiceImpl 伪调度**: `ai-domain/route-optimizer/.../RouteOptimizerDomainServiceImpl.java` — 最近邻算法仅按 sequence 排序，无真实路径优化逻辑
- [ ] **[Critical] RiskControlServiceImpl 风控失效**: `payment-service/.../RiskControlServiceImpl.java` — `isBlacklistedAddress()` 始终返回 false，风控形同虚设
- [ ] **[Critical] BlockchainServiceStubImpl 伪链交互**: `blockchain-domain/.../BlockchainServiceStubImpl.java` — 所有方法仅打印日志后返回 null/0/空集合，无真实区块链交易
- [ ] **[Critical] MinioStorageService.getFileContent 未实现**: `common/.../MinioStorageService.java` — `getFileContent` 方法 throw `UnsupportedOperationException`
- [ ] **[Critical] GraphQLDataProvider 购物车查询抛异常**: `gateway/.../GraphQLDataProvider.java` — 3 个购物车相关方法直接 throw `UnsupportedOperationException("GraphQLDataProvider")`
- [ ] **[Critical] LiveService ProductClient 返回 null**: `live-service/.../ProductClient.java` — `getProductById()` 始终返回 `Mono.empty()`，导致直播关联商品查询全部失效

#### Critical — 异常吞噬

- [ ] **[Critical] Gateway UserClient 异常被吞**: `gateway/.../UserClient.java` — 多处 `try-catch` 仅记录日志后返回 null/空，不抛异常，导致上游无法感知下游失败
- [ ] **[Critical] PromotionServiceRpcImpl NumberFormatException 被吞**: `promotion-service/.../PromotionServiceRpcImpl.java:127` — 解析数字失败时仅 `log.warn` 后 return，不抛异常

#### Medium — System.err.println / printStackTrace 替代日志

- [ ] **[Medium] Common 模块多处使用 System.err.println**: `common/.../LogQueryService.java`、`AlertService.java` — 应改为 logger.error
- [ ] **[Medium] Common 模块多处使用 e.printStackTrace()**: `common/.../TelegramAuth.java`、`SecretUtilsKey.java`、`OAuth1Utils.java` — 应改为 logger.error
- [ ] **[Medium] Common 模块多处使用 System.out.println**: `common/.../AlgorithmUtils.java`、`LogRocksDBAppender.java` — 应改为 logger.info
- [ ] **[Medium] Gateway CircuitBreakerFilter System.err.println**: `gateway/.../CircuitBreakerFilter.java` — 应改为 logger.warn

#### Medium — 代码规范问题

- [ ] **[Medium] @Deprecated 无替代说明**: `factory-domain/mes-service/.../CodeRule.java:151`、`payment-service/.../ReconciliationServiceImpl.java:71` — @Deprecated 注解应标注 `forRemoval` 和 `since` 参数
- [ ] **[Medium] UserService.updateUser 默认实现仅为占位符**: `user-service/.../UserService.java:42` — interface default 方法，直接 return 0，无真实逻辑
- [ ] **[Medium] PasskeyServiceImpl.encodePublicKey 始终返回空串**: `user-service/.../PasskeyServiceImpl.java:160` — 密钥编码方法 return ""，无法用于真实密钥交换
- [ ] **[Medium] DecisionServiceImpl 日志冗余**: `payment-service/.../DecisionServiceImpl.java:158` — `e.printStackTrace()` + `log.error` 同时使用，应统一

#### Medium — TODO 残留

- [ ] **[Medium] ProductService 核心方法仅含 TODO**: `product-service/.../ProductService.java` — `createProduct()` 和 `updateProduct()` 方法体仅包含 `// TODO` 注释，无任何实现逻辑
- [ ] **[Medium] PaymentRpcService 统计接口返回硬编码 0**: `payment-service/.../PaymentRpcService.java` — `getPaymentStatistics()` 和 `getDailyPaymentStats()` 所有数值返回 0
- [ ] **[Medium] ApiDocumentationService 订阅过滤 TODO**: `platform-domain/developer-portal-service/.../ApiDocumentationService.java:398` — 订阅状态过滤逻辑留空

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

