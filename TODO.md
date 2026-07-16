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

- [x] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [x] 创建 `crm-service` 模块（基于 backend-api 父 POM）
   - [x] 定义领域实体：`Lead`, `Opportunity`, `SalesPipeline`, `CustomerServiceTicket`, `Campaign`, `Contact`
   - [x] 实现销售管道：商机创建 → 阶段推进 → 赢单/输单
   - [x] 实现客户服务工单：创建 → 分配 → 处理 → 关闭
   - [x] 实现与用户服务集成：客户数据同步（UserServiceClient + /api/crm/leads/sync/* 端点）
   - [x] 添加 REST API + GraphQL 端点（Apollo Federation 子图，POST /graphql）
   - [x] 添加 Protobuf 定义：`protos/erp/CrmService.proto`
   - [x] 添加后台管理页面：`views/crm/*` 系列页面

- [x] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [x] 基于现有 `reporting-service` 扩展：添加 OLAP 聚合查询能力（OLAP 实现在 data-pipeline 服务）
   - [x] 实现销售分析看板：销量趋势、品类分布、区域对比（对接 data-analysis-service）
   - [x] 实现财务分析看板：收入/成本/利润趋势、预算执行率（对接 data-analysis-service）
   - [x] 实现库存分析看板：周转率、ABC分析、安全库存预警（对接 data-analysis-service）
   - [x] 实现生产分析看板：OEE、良品率、计划达成率（data-pipeline 新增 PRODUCTION 域 + /api/analytics/production 端点）
   - [x] 添加后台 BI 看板页面：`views/bi/*` 系列页面

- [x] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [x] ERP 生产订单发布时发送领域事件(MQ/Kafka)到 MES
   - [x] MES 工单报工完成后发送完工事件到 ERP
   - [x] 财务成本核算模块监听完工事件自动归集成本
   - [x] 实现端到端集成测试

### [技术栈违规：部分模块使用了 JPA/MySQL 而非 MyBatis Plus/PostgreSQL]

- [ ] **[Critical] forecasting-service** — 3 个实体使用 JPA `@Entity`/`@Id`/`@Column` 注解；pom.xml 含 `spring-boot-starter-data-jpa` + `mysql-connector-j`。需改为 MyBatis Plus + PostgreSQL。
- [ ] **[Critical] route-optimizer** — 3 个实体使用 JPA `@Entity`/`@Id`/`@Column` 注解；pom.xml 含 `spring-boot-starter-data-jpa`。需改为 MyBatis Plus + PostgreSQL。
- [ ] **[Critical] developer-portal-service** — 5 个实体使用 JPA `@Entity`/`@Id`/`@Column` 注解；pom.xml 含 `spring-boot-starter-data-jpa` + `mysql-connector-j`。需改为 MyBatis Plus + PostgreSQL。
- [ ] **[Medium] data-analysis-service** — pom.xml 含 `spring-boot-starter-data-jpa` + `mysql-connector-j`，但代码未使用 JPA。需清理 pom.xml 中多余依赖。
- [ ] **[Medium] data-pipeline** — pom.xml 含 `spring-boot-starter-data-jpa`，但实体已使用 MyBatis Plus `@TableName`。需清理 pom.xml 中多余依赖。
- [ ] **[Medium] traceability-service** — pom.xml 含 `spring-boot-starter-data-jpa`，但实体已使用 MyBatis Plus `@TableName`。需清理 pom.xml 中多余依赖。

### [Digital Twin UI 交互流程未完整恢复]（新任务）

- [ ] **[Digital Twin] 右侧面板 Tab 与告警/通知/SCADA/追溯功能链路未完整闭环**
  - [ ] 在 `apps/digital-twin/system-management/src/renderer/pages/DigitalTwinPage.tsx` 中确认并恢复完整 UI 流程：
    - [ ] ToastContainer（告警/动作反馈）渲染挂载
    - [ ] 右侧 Tab：规则配置（rules）、SCADA（scada）、追溯（trace）三块的入口与状态切换
    - [ ] rules Tab：`AlertRuleList` 规则配置列表能正常加载与分页/编辑（若存在交互）
    - [ ] scada Tab：`ScadaPanel` 能基于选中设备展示遥测与指令下发
    - [ ] trace Tab：`TraceabilityPanel` 能展示完整链路并支持正向/反向追溯
    - [ ] 设备未选中时的占位逻辑从"点击提示"改为可指导的体验（并确保不会影响三块 Tab 的挂载）
    - [ ] charts Tab 的声音/通知开关（soundEnabled/notifEnabled）与 `AudioMonitor`/通知告警触发的接入
  - [ ] 校验环境变量/端口提示：异常文案应基于 `DIGITAL_TWIN_API_BASE_URL` 推导端口，而非写死 10102。

### [扫描发现的虚假/占位符/内存实现]（已修复并确认通过）

基于 `server/` 全量 Java 源代码扫描发现的虚假/占位符实现，以下项目已全部修复并通过代码规范检查：

- [x] **[Critical] AI 域 RouteOptimizer 内存 Repository** — 改为 JPA 持久化（JpaRepository + @Entity）
- [x] **[Critical] AI 域 ForecastingService 内存 Repository** — 改为 JPA 持久化
- [x] **[Critical] Blockchain WalletService 内存 Repository** — 改为 MyBatis Plus 持久化
- [x] **[Critical] SupplyChain 库存盘点内存 Repository** — 改为 MyBatis Plus 持久化
- [x] **[Critical] Common AuditLogRepository** — 改为 MyBatis Plus 持久化
- [x] **[Critical] ForecastingDomainServiceImpl 伪训练** — 改为基于真实历史数据的交叉验证精度计算
- [x] **[Critical] RouteOptimizerDomainServiceImpl 伪调度** — 实现 Haversine 距离+容量约束+时间窗算法
- [x] **[Critical] RiskControlServiceImpl 风控失效** — 实现真实黑名单 HashSet 查找
- [x] **[Critical] BlockchainServiceStubImpl 伪链交互** — 实现本地内存钱包状态跟踪
- [x] **[Critical] MinioStorageService.getFileContent 未实现** — 实现 MinioClient.getObject() 调用
- [x] **[Critical] GraphQLDataProvider 购物车查询抛异常** — 实现 addToCart/removeFromCart/clearCart
- [x] **[Critical] LiveService ProductClient 返回 null** — 实现内存商品存储和真实查询
- [x] **[Critical] Gateway UserClient 异常被吞** — try-catch 改为重新抛出 RuntimeException
- [x] **[Critical] PromotionServiceRpcImpl NumberFormatException 被吞** — 改为抛出 IllegalArgumentException
- [x] **[Medium] Common 模块 System.err.println** — 改为 logger.error
- [x] **[Medium] Common 模块 e.printStackTrace()** — 改为 logger.error
- [x] **[Medium] Common 模块 System.out.println** — 改为 logger.info/addInfo
- [x] **[Medium] Gateway CircuitBreakerFilter System.err.println** — 改为 logger.warn
- [x] **[Medium] @Deprecated 无替代说明** — 添加 forRemoval 和 since 参数
- [x] **[Medium] UserService.updateUser 占位符** — 改为抛出 UnsupportedOperationException
- [x] **[Medium] PasskeyServiceImpl.encodePublicKey 空串** — 实现 Base64 编码
- [x] **[Medium] DecisionServiceImpl 日志冗余** — 移除 e.printStackTrace()
- [x] **[Medium] PaymentRpcService 统计接口硬编码 0** — 注入 ExchangeOrderRepository 实现真实数据库查询
- [x] **[Medium] ApiDocumentationService 订阅过滤 TODO** — 实现订阅状态过滤逻辑


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

