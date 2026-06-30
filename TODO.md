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

#### MES 领域 (高优先级)

- [ ] **[MES] 有限产能排程 (Finite Capacity Scheduling)**: `server/factory-domain/mes-service` 中添加排程引擎 [代码质量不达标: 异常被吞, 前端文件超长, 缺少单元测试]
   - [x] 定义排程数据模型：`ScheduleOrder`, `ScheduleResource`, `ScheduleResult` (含内嵌 `ScheduleOperation`, `TimeSlot`, `ScheduleConflict`) [需修复 SchedulingDomainServiceImpl 异常吞并问题]
   - [x] 实现基础排程算法：前向排程(Forward Scheduling)、后向排程(Backward Scheduling) — `SchedulingDomainServiceImpl` [需修复异常吞并: 第78行和第141行catch Exception后仅记录日志未重新抛出]
   - [x] 实现约束检查：资源可用性 (`ScheduleResource.isAvailable()`), 冲突报告 (`ScheduleConflict`)
   - [x] 添加调度 REST API：`/api/mes/scheduling/orders|resources|forward|backward`
   - [x] 添加后台调度管理页面：`views/mes/scheduling/index.vue` (工单列表+资源管理.含前向/后向排程按钮,工序展开,调度结果报警) [需拆分为多个组件, 当前989行超出500行限制]

- [ ] **[MES] 人员/工时跟踪 (Labor & Time Tracking)**: 记录操作员资质、工时、考勤及人员到工作中心的分配 [代码质量不达标: 缺少单元测试]
   - [x] 定义领域实体：`Operator`, `OperatorSkill`, `WorkCenterAssignment`, `TimeRecord`, `Attendance`
   - [x] 实现工时记录：员工报工、工时核算、产出记录 (clockIn/Out, submit/approve flow)
   - [x] 实现工作中心分配：人员 → 工位/产线绑定
   - [x] 添加 REST API：`/api/mes/labor/operators|attendance|time-records|assignments`
   - [x] 添加后台管理页面：`views/mes/labor/index.vue` (含 4 个 Tab: 操作员/考勤/工时/分配)

- [ ] **[MES] 制造可追溯性完善 (Manufacturing Genealogy)**: 在已有 TraceModel/TraceRecord 基础上补全"制造族谱" [代码质量不达标: 缺少单元测试]
   - [x] 完善溯源数据模型：连接成品 → 批次原料 → 设备 → 操作员 → 工艺参数
   - [x] 实现自动收集：工单完工时自动关联原料批次、设备、操作员
   - [x] 添加批次追溯查询 API：正向(原料→成品)和反向(成品→原料)
   - [x] 添加后台追溯查询页面：`views/mes/traceability/index.vue`

- [ ] **[MES] 实时数据采集集成 (SCADA Integration)**: 基于已有 Equipment.mqttTopic 字段构建完整数据采集层 [代码质量不达标: 异常吞并]
   - [x] 实现 MQTT 设备数据消费者：订阅设备 topic，解析遥测数据 (`MqttTelemetrySubscriber`, `MqttTelemetryService`)
   - [x] 定义采集数据模型：`TelemetryRecord`, `TelemetryMetric`, `DeviceCommand`
   - [x] 实现设备命令下发：从 MES 下发参数配置到设备 (`MqttCommandPublisher`, `ScadaDomainService.dispatchCommand`)
   - [x] 实现实时状态看板：设备 OEE、生产进度、异常告警 (SCADA 监控后台页面 + REST API)

**数字孪生MES集成** [代码质量不达标: 前端编译错误, 异常吞并]:
- [ ] **[验收] 数字孪生 MES API 连通性验证**: `apps/digital-twin/system-management/src/renderer/services/mes-api.ts` — 确认 mesApi 各方法能正确调用 `mes-service` 的 SCADA/追溯端点 [getTraceChain 存在异常吞并, 需修复]
- [ ] **[验收] SCADA 面板设备遥测实时展示**: `apps/digital-twin/system-management/src/renderer/components/digital-twin/scada/ScadaPanel.tsx` — 验证左侧选中设备后，面板展示实时遥测指标（温度/压力/转速等），超限值红色告警
- [ ] **[验收] SCADA 指令下发与响应**: `ScadaPanel.tsx` — 验证指令类型选择 + JSON 参数发送后，最近指令列表能显示状态变化（PENDING→SENT→EXECUTED/FAILED）
- [ ] **[验收] 追溯面板完整链查询**: `apps/digital-twin/system-management/src/renderer/components/digital-twin/traceability/TraceabilityPanel.tsx` — 验证输入追溯码后完整链视图展示根节点 + 正向路径 + 反向路径
- [ ] **[验收] 追溯面板正向/反向追溯**: `TraceabilityPanel.tsx` — 验证单独正向/反向追溯按钮返回正确节点列表
- [ ] **[验收] 数字孪生标签页集成**: `apps/digital-twin/system-management/src/renderer/pages/DigitalTwinPage.tsx` — 验证右侧面板新增 SCADA / 追溯标签页正常切换渲染，无白屏或 JS 错误
- [ ] **[验收] MES API 环境变量配置**: `apps/digital-twin/system-management/vite.config.ts` + `.env.example` — 验证 `VITE_MES_API_URL/HOST/PORT` 注入正确，数字孪生启动时可连接 mes-service
- [ ] **[安全] SCADA 指令鉴权**: 设备指令下发端点 `/api/mes/scada/commands` 需确认已集成 Spring Security 权限校验，防止未授权操作产线设备
- [ ] **[安全] 追溯数据访问控制**: 追溯链查询接口需校验用户权限，防止越权访问非授权产品的批次追溯数据

#### 编译错误修复 (新增 - 紧急)

- [ ] **[紧急] 修复前端TypeScript编译错误**: apps/backstage-admin 构建失败
   - [ ] 修复 `src/locales/en-US.ts(774,7)` 和 `src/locales/en-US.ts(775,7)` 及 `src/locales/zh-CN.ts(835,7)` 和 `src/locales/zh-CN.ts(836,7)` 中重复的属性名 (error TS1117: An object literal cannot have multiple properties with the same name)
   - [ ] 修复 `src/views/mes/scheduling/index.vue(592,29)` 类型不匹配: createResource 参数类型错误
   - [x] 修复 `src/views/recommendation/index.vue(118,5)` RecommendationStatistics 类型缺失必要属性

#### 供应链领域 (中优先级)

- [ ] **[Supply Chain] 退货管理 (RMA - Return Material Authorization)**: 逆向物流模块
   - [ ] 定义领域实体：`RmaOrder`, `RmaOrderItem`, `RmaInspection`, `RmaDisposition`, `ReturnShipping`
   - [ ] 实现退换货流程：退货申请 → 质检 → 处理决定(退款/换货/维修) → 返仓/报废
   - [ ] 实现与库存服务集成：RMA 质检通过后自动触发入库
   - [ ] 实现与结算服务集成：退款自动触发结算
   - [ ] 添加 REST API + Protobuf 定义
   - [ ] 添加 Protobuf 定义：`protos/supply-chain/RmaService.proto`
   - [ ] 添加后台管理页面：`views/rma/*` 系列页面

- [ ] **[Supply Chain] 分布式订单管理 (DOM)**: 跨仓库订单承诺、寻源和履行
   - [ ] 定义领域实体：`DomOrder`, `DomOrderLine`, `FulfillmentPlan`, `SourcingRule`
   - [ ] 实现订单寻源：根据库存、距离、成本自动选择最优仓库
   - [ ] 实现可用量承诺(ATP)检查
   - [ ] 添加 REST API
   - [ ] 添加后台管理页面

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

# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

