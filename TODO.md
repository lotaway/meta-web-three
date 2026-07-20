# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### [Backend Admin Missing]

The following backend services have been created, but lack corresponding admin and operation pages. Each needs to be added:

- **mall-domain** (13 services: after-sale, cart, group-buying, live, mall-supplier, order, payment, product, promotion, recommendation, review, risk-control, user) — only order(oms), product(pms), promotion, recommendation, review, risk-control have pages. Missing: cart, group-buying, live, mall-supplier, payment, user (~6 missing)
- **ai-domain** (3 services: ai-warehouse, forecasting, route-optimizer) — only forecasting + routing have pages. Missing: ai-warehouse
- **factory-domain / mes-service** (3 services: digital-twin, mes, production) — only mes has pages. Missing: digital-twin, production
- **blockchain-domain** (2 services: traceability, wallet) — ✅ **Both have pages** under `/blockchain`
- **erp-domain** (7 services: crm, finance, hrm, invoice, project, reporting, settlement) — only crm, hrm, settlement, reporting have pages. Missing: finance, invoice, project
- **platform-domain** (8 services: commission, cs, data-analysis, developer-portal, media, message, social-commerce, user-action) — only cs has pages. Missing: commission, data-analysis, developer-portal, media, message, social-commerce, user-action (~7 missing)
- **supply-chain-domain** (8 services: dom, inventory-alert, inventory, logistics, procurement, rma, supplier, warehouse) — only dom, inventory(incl. alert), logistics, rma, supplier have pages. Missing: procurement, warehouse

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

