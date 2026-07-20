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

#### ERP 领域 (中优先级)

- [x] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [x] **应用层依赖 MyBatis Plus**：已修复（前一阶段）
   - [x] **零单元测试**：已为 LeadCommandService、OpportunityCommandService、TicketCommandService、CampaignQueryService 编写 29 个单元测试

- [x] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [x] **sendEmail() 为伪代码**：已替换为真实 JavaMailSender，添加 `spring-boot-starter-mail` 依赖
   - [x] **DDD 分层违规**：已将毛利率计算等从 SalesReportQueryService 提取到新创建的 SalesReportDomainService（领域层）
   - [x] **领域层使用 Lombok**：InventoryReport 已移除 @Builder/@Getter/@ToString，改为手动 getter/toString + 静态工厂方法
   - [x] **零单元测试**：已为 SalesReportDomainService、SalesReportQueryService、ReportDeliveryService 编写测试
   - [x] **前端缺少库存/生产专用页面**：已创建 inventory.vue 和 production.vue，并在 router 中添加对应路由

- [x] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [x] **通过修改现有代码实现**：已从 WorkOrderCommandService 移除 notifyWorkOrderCompleted（应由事件监听器处理）和 prepareSplit/saveSplitOrders（已由 WorkOrderSplitService 覆盖）
   - [x] **零单元测试**：已为 MesDomainServiceImpl（13 个测试）、ProductionEventProcessorImpl（3 个测试）、MesCrossDomainEventPublisher（3 个测试）编写共计 19 个单元测试

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

