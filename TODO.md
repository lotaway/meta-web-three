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
   - [ ] **已修复**：FE/BE API 协议不匹配（改为 @RequestBody DTO），库存/结算集成（添加 EventListener + Stub 服务），硬编码值（提取为常量），函数返回值+副作用混合（6 个领域方法已拆分）
   - [ ] **仍存在问题**：
     - [ ] **Protobuf 枚举全部为空**：6 个枚举只有 UNSPECIFIED=0，无实际值
     - [ ] **函数超 20 行**：`createRma()` 30 行，`toOrderDTO()` 34 行，`recordInspection()` 23 行
     - [ ] **零单元测试**：核心退换货流程无任何测试
     - [ ] **前端对话框固定宽度无响应式**：4 个对话框使用固定 px 宽度

- [ ] **[Supply Chain] 分布式订单管理 (DOM)**: 跨仓库订单承诺、寻源和履行
   - [ ] **已修复**：FE/BE 类型不匹配（`row.id` 替换 `row.domOrderNo`），字段名不一致（`items`→`lines`），状态值不匹配（移除无效枚举），分页未实现（添加 Page 查询），硬编码值（提取常量）
   - [ ] **仍存在问题**：
     - [ ] **函数超 20 行**：`createDomOrder()` 52 行（超 32 行）
     - [ ] **领域层引入 Spring 注解**：`DomSourcingProperties` 在 domain 包中使用 `@Component`/`@ConfigurationProperties`
     - [ ] **缺少输入校验**：Controller 方法缺少 `@Valid`，请求体字段无校验注解
     - [ ] **隐式共享状态**：`DomSequenceGeneratorImpl` 静态 `AtomicLong` + `WarehouseServiceMockClient` 静态 Map
     - [ ] **死代码**：`infrastructure/rpc/` 下 3 个文件与 `domain/service/` 完全重复
     - [ ] **零单元测试**：核心 ATP 检查、寻源算法无测试

#### ERP 领域 (中优先级)

- [ ] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [ ] **已修复**：FE/BE API 路径（单数→复数），HTTP 方法（assignTicket POST→PUT），参数传递（body→params/去除 body），异常被吞（UserServiceClient 改为抛出 Runtime），硬编码字符串（提取常量），前端 catch 块（添加 ElMessage.error）
   - [ ] **仍存在问题**：
     - [ ] **文件超 500 行**：`CrmSubgraphConfig.java` 550 行
     - [ ] **领域层依赖 MyBatis Plus**：所有 Repository `extends BaseMapper<T>` 引入 ORM 到领域层
     - [ ] **应用层依赖 MyBatis Plus**：`LeadQueryService`/`OpportunityQueryService`/`TicketQueryService` 导入 `LambdaQueryWrapper`
     - [ ] **DDD 分层违规**：`ContactController`/`CampaignController` 直接注入 Repository 绕过应用层
     - [ ] **缺少输入校验**：GraphQL 控制器对 `env.getArgument("id")` 无 null 检查
     - [ ] **零单元测试**：无 `src/test/` 目录

- [ ] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [ ] **已修复**：PRODUCTION 域（添加表/维度/指标映射，改用 OLAP 服务），硬编码值（重命名为描述性常量名），异常被吞（sendEmail 改为抛出），前端错误反馈（添加 ElMessage.error）
   - [ ] **仍存在问题**：
     - [ ] **函数超 20 行**：`appendSalesReportContent()` 37 行，`appendFinancialReportContent()` 45 行，`bfsImpact()` 28 行
     - [ ] **DDD 分层违规**：`application/query/` 服务包含领域逻辑（毛利率计算、报表聚合）
     - [ ] **领域层使用 Lombok**：`@Builder`/`@Getter`/`@ToString` 在领域实体中
     - [ ] **缺少输入校验**：OLAP/报表 API 无 `@Valid` 或参数校验
     - [ ] **使用已废弃 API**：`BigDecimal.ROUND_HALF_UP` (Java 9+ 废弃)
     - [ ] **零单元测试**：无测试覆盖
     - [ ] **前端 API 路径不匹配**：`getInventoryTurnover()` 调用 `/api/v1/analysis/inventory/overview` 但后端为 `/api/analytics/inventory`
     - [ ] **前端缺少库存/生产专用页面**：TODO 要求 views/bi/* 系列页面，但库存和生产嵌入在 index.vue 标签页
     - [ ] **前端无响应式**：`el-col :span="6"` 无 `:xs`/`:sm` 断点，手机端溢出

- [ ] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [ ] **已修复**：异常被吞（改为抛出 RuntimeException），领域层依赖 Spring（移除 `extends ApplicationEvent`），隐式共享状态（WorkOrder static→instance），事件未显式建模（Map→typed records），事件名用字符串（改用 EventType 枚举），日志违规（debug→info），接口命名不一致（Processor/EventListener 统一）
   - [ ] **仍存在问题**：
     - [ ] **硬编码 TOPIC/Kafka 配置**：topic 名、JSON 格式字符串全部硬编码
     - [ ] **函数超 20 行**：集成测试 2 个函数分别 33 行和 26 行
     - [ ] **通过修改现有代码实现**：`WorkOrderCommandService` 被大幅重构，添加了非原有职责的方法
     - [ ] **零单元测试**：`MesDomainServiceImpl`、`ProductionEventProcessor`、`MesCrossDomainEventPublisher` 均无测试

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

