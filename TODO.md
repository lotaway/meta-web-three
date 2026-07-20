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

- [ ] **[ERP] 客户关系管理 (CRM)**: 销售管道、商机跟踪、客户服务工单、营销活动
   - [ ] **零单元测试**：无 `src/test/` 目录

- [ ] **[ERP] BI 商业智能与分析层**: 仪表板、即席查询、数据可视化
   - [ ] **sendEmail() 为伪代码**：仅记录日志但未真正调用邮件发送 API（无 JavaMailSender），emailEnabled=false 时静默跳过，应接入真实邮件发送实现
   - [ ] **DDD 分层违规**：`application/query/` 服务包含领域逻辑（毛利率计算、报表聚合）
   - [ ] **领域层使用 Lombok**：`@Builder`/`@Getter`/`@ToString` 在领域实体中
   - [ ] **零单元测试**：无测试覆盖
   - [ ] **前端缺少库存/生产专用页面**：TODO 要求 views/bi/* 系列页面，但库存和生产嵌入在 index.vue 标签页

- [ ] **[Cross] ERP-MES 数据闭环集成**: 打通 ERP 生产订单 → MES 报工 → 财务成本核算
   - [ ] **通过修改现有代码实现**：`WorkOrderCommandService` 被大幅重构，添加了非原有职责的方法
   - [ ] **零单元测试**：`MesDomainServiceImpl`、`ProductionEventProcessor`、`MesCrossDomainEventPublisher` 均无测试

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

