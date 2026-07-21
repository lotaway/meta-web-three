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

- [ ] **CRM** — 后端有 12 处 `Map<String, Object>` 类型侵蚀（UserServiceClient、LeadController、ContactController、GraphQL），违反"禁止接口层泄露实现细节"。建议：为 UserService 客户端和 GraphQL 定义专用 DTO。
- [ ] **BI/商业智能** — 前端 15 处 `as any` 强制类型转换（views/bi/ 下 5 个文件）、4 处 `Record<string, any>`（bi.ts），违反 TypeScript 类型安全。建议：为所有 API 响应定义完整接口，替换 as any。
- [ ] **资产管理（固定资产）** — 后端 FixedAssetController 10 处 `Map.of()` 硬编码响应 + 12 处 `Map<String, Object>` 类型侵蚀，违反"禁止硬编码魔法数字"和"禁止接口层泄露实现细节"。建议：为 create/update/delete/transfer 等操作定义专用 Response DTO。
- [ ] **预算管理** — 后端 BudgetController 11 处 `Map.of()` 硬编码响应 + 11 处 `Map<String, Object>` 类型侵蚀，前端 1 处 `Record<string, any>`。建议：定义专用 Response DTO 替换 Map.of。

# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

