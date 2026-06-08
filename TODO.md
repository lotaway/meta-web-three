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


### Recommendation Service Problem

- [x] 检查[智能推荐系统](server/mall-domain/recommendation-service/)是否已经实现完善，对应到[前端](apps/client/)和[后台](apps/backstage-admin/) (AI推荐算法，基于用户行为个性化推荐商品)，智能推荐系统已实现完善但存在以下待修复问题：
- [x] **[严重] 前端行为追踪链路未接通**: `recordBehavior`、`markClicked`、`markPurchased` mutation 在前端已定义但未被任何组件调用（`apps/client/app/(tabs)/index.tsx` 商品曝光/点击、`apps/client/app/product/[id].tsx` 详情页浏览/加购、`apps/client/app/components/product/RelatedProducts.tsx` 推荐点击/购买均需接入）
- [x] **[严重] 首页推荐硬编码 userId=1**: `apps/client/containers/home/HomeContainer.tsx:61` 写死 `userId=1`，所有用户看到相同的推荐结果，应改为从 auth context 获取当前登录用户ID
- [x] **[中] 测试覆盖严重不足**: 仅1个测试文件10个测试方法（仅覆盖 `RecommendationDomainServiceImpl`），缺少5种算法单元测试、应用层测试、GraphQL DataFetcher 测试、仓储层集成测试、控制器集成测试
- [x] **[中] 缺少 Dockerfile**: 与 `user-service`、`product-service`、`order-service`、`payment-service` 等同类服务相比，推荐服务无法容器化部署
- [x] **[中] 缺少离线计算/定时任务**: 流行度推荐依赖预计算结果但无 `@Scheduled` 自动计算；商品相似度矩阵需手动触发更新，无自动定时更新机制
- [x] **[低] 后台管理功能冗余**: `src/views/recommendation/index.vue`（规则类型 BOOST/FILTER/RE_RANK/EXCLUDE）与 `src/views/sms/recommendation/index.vue`（规则类型 COLLABORATIVE/CONTENT_BASED/HYBRID/POPULARITY）两个推荐规则管理入口功能重叠，建议统一合并
- [x] **[低] 后台缺少推荐效果数据看板**: 后端 `/api/admin/recommendation/statistics` 已提供统计数据，但后台管理缺少对应数据展示页面（CTR趋势图、转化率、推荐覆盖率等）

### [Pending Features]

- [ ] 添加 GraphQL Federation 统一查询层 (整合多个微服务的GraphQL端点，提供统一API)
   - [ ] 需要端到端运行时验证：启动所有子图服务 + gateway，用 GraphQL 客户端请求跨子图联合查询（如通过 cart 查 product），验证 Federation 路由和实体解析正确

---

### [ERP/MES/SupplyChain 缺失模块实现]

基于 `TODO_ERP.md` 分析对比现有代码后，确认以下真正缺失的功能模块，按优先级排列：

#### MES 领域 (高优先级)

- [x] **[MES] 有限产能排程 (Finite Capacity Scheduling)**: `server/factory-domain/mes-service` 中添加排程引擎
   - [x] 定义排程数据模型：`ScheduleOrder`, `ScheduleResource`, `ScheduleResult` (含内嵌 `ScheduleOperation`, `TimeSlot`, `ScheduleConflict`)
   - [x] 实现基础排程算法：前向排程(Forward Scheduling)、后向排程(Backward Scheduling) — `SchedulingDomainServiceImpl`
   - [x] 实现约束检查：资源可用性 (`ScheduleResource.isAvailable()`), 冲突报告 (`ScheduleConflict`)
   - [x] 添加调度 REST API：`/api/mes/scheduling/orders|resources|forward|backward`
   - [x] 添加后台调度管理页面：`views/mes/scheduling/index.vue` (工单列表+资源管理.含前向/后向排程按钮,工序展开,调度结果报警)

- [x] **[MES] 人员/工时跟踪 (Labor & Time Tracking)**: 记录操作员资质、工时、考勤及人员到工作中心的分配
   - [x] 定义领域实体：`Operator`, `OperatorSkill`, `WorkCenterAssignment`, `TimeRecord`, `Attendance`
   - [x] 实现工时记录：员工报工、工时核算、产出记录 (clockIn/Out, submit/approve flow)
   - [x] 实现工作中心分配：人员 → 工位/产线绑定
   - [x] 添加 REST API：`/api/mes/labor/operators|attendance|time-records|assignments`
   - [x] 添加后台管理页面：`views/mes/labor/index.vue` (含 4 个 Tab: 操作员/考勤/工时/分配)

- [x] **[MES] 制造可追溯性完善 (Manufacturing Genealogy)**: 在已有 TraceModel/TraceRecord 基础上补全"制造族谱"
    - [x] 完善溯源数据模型：连接成品 → 批次原料 → 设备 → 操作员 → 工艺参数
    - [x] 实现自动收集：工单完工时自动关联原料批次、设备、操作员
    - [x] 添加批次追溯查询 API：正向(原料→成品)和反向(成品→原料)
    - [x] 添加后台追溯查询页面：`views/mes/traceability/index.vue`

- [x] **[MES] 实时数据采集集成 (SCADA Integration)**: 基于已有 Equipment.mqttTopic 字段构建完整数据采集层
   - [x] 实现 MQTT 设备数据消费者：订阅设备 topic，解析遥测数据 (`MqttTelemetrySubscriber`, `MqttTelemetryService`)
   - [x] 定义采集数据模型：`TelemetryRecord`, `TelemetryMetric`, `DeviceCommand`
   - [x] 实现设备命令下发：从 MES 下发参数配置到设备 (`MqttCommandPublisher`, `ScadaDomainService.dispatchCommand`)
   - [x] 实现实时状态看板：设备 OEE、生产进度、异常告警 (SCADA 监控后台页面 + REST API)

**数字孪生MES集成**:
- [x] **[验收] 数字孪生 MES API 连通性验证**: `apps/digital-twin/system-management/src/renderer/services/mes-api.ts` — 确认 mesApi 各方法能正确调用 `mes-service` 的 SCADA/追溯端点
- [x] **[验收] SCADA 面板设备遥测实时展示**: `apps/digital-twin/system-management/src/renderer/components/digital-twin/scada/ScadaPanel.tsx` — 验证左侧选中设备后，面板展示实时遥测指标（温度/压力/转速等），超限值红色告警
- [x] **[验收] SCADA 指令下发与响应**: `ScadaPanel.tsx` — 验证指令类型选择 + JSON 参数发送后，最近指令列表能显示状态变化（PENDING→SENT→EXECUTED/FAILED）
- [x] **[验收] 追溯面板完整链查询**: `apps/digital-twin/system-management/src/renderer/components/digital-twin/traceability/TraceabilityPanel.tsx` — 验证输入追溯码后完整链视图展示根节点 + 正向路径 + 反向路径
- [x] **[验收] 追溯面板正向/反向追溯**: `TraceabilityPanel.tsx` — 验证单独正向/反向追溯按钮返回正确节点列表
- [x] **[验收] 数字孪生标签页集成**: `apps/digital-twin/system-management/src/renderer/pages/DigitalTwinPage.tsx` — 验证右侧面板新增 SCADA / 追溯标签页正常切换渲染，无白屏或 JS 错误
- [x] **[验收] MES API 环境变量配置**: `apps/digital-twin/system-management/vite.config.ts` + `.env.example` — 验证 `VITE_MES_API_URL/HOST/PORT` 注入正确，数字孪生启动时可连接 mes-service
- [x] **[安全] SCADA 指令鉴权**: 设备指令下发端点 `/api/mes/scada/commands` 需确认已集成 Spring Security 权限校验，防止未授权操作产线设备
- [x] **[安全] 追溯数据访问控制**: 追溯链查询接口需校验用户权限，防止越权访问非授权产品的批次追溯数据

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

# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)