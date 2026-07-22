# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### [GitHub Issues]

- [ ] **[#1] Solana 商城模板集成**: Token, NFT, SFT 创建与管理；Token 作为商品销售；活动和佣金功能
  - [ ] Token/NFT/SFT 创建和管理合约
  - [ ] Token 作为商品销售的商城前端集成
  - [ ] 活动与佣金功能
  - 链接: https://github.com/lotaway/meta-web-three/issues/1

## ERP

### 预算管理
- [x] BudgetController — 11 处 `Map.of("success", true)` / `Map.of("id", id, "success", true)` 替换为 `ResponseEntity<Void>` 或 `IdResponse` DTO
- [x] budget/index.vue — 1 处 `Record<string, any>` 替换为 `ListBudgetsParams` 接口

### 资产管理（固定资产）
- [x] FixedAssetController — 10 处 `Map.of("success", true)` / `Map.of("id", id, "success", true)` 替换为 `ResponseEntity<Void>` 或 `IdResponse` DTO
- [x] FixedAssetQueryService — `getAssetStatistics()` / `getDepreciationStatistics()` / `getInventoryStatistics()` 返回 `Map<String, Object>` 替换为 `*Statistics` record
- [x] asset/card/index.vue — 1 处 `as any`（updateAsset 调用）替换为类型安全的方式
- [x] asset/inventory/index.vue — 1 处 `as any`（formData）替换为正确类型

### CRM
- [x] UserServiceClient.java — 6 处 `Map<String, Object>` 替换为 `UserDTO` / `UserStatsDTO`
- [x] LeadController.java — 2 处 `Map<String, Object>` 返回替换为 `UserDTO`
- [x] ContactController.java — 1 处 `Map<String, Object>` 返回替换为 `UserDTO`
- [x] CrmSubgraphController.java — GraphQL `Map<String, Object>` 输入/输出替换为 `GraphQLRequest` / `GraphQLResponse` DTO

### BI/商业智能
- [x] bi.ts — 4 处 `Record<string, any>` 替换为 `ProductionAnalyticsResponse` / `SalesFunnelResponse` 接口
- [x] bi/index.vue — 多处 `as any` 替换为类型安全方式
- [x] bi/sales.vue — 多处 `as any` 替换为类型安全方式
- [x] bi/financial.vue — 1 处 `as any` 替换为类型安全方式
- [x] bi/inventory.vue — 多处 `as any` 替换为类型安全方式
- [x] bi/production.vue — 多处 `as any` 替换为类型安全方式

# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

