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

- [ ] 检查[智能推荐系统](server/mall-domain/recommendation-service/)是否已经实现完善，对应到[前端](apps/client/)和[后台](apps/backstage-admin/) (AI推荐算法，基于用户行为个性化推荐商品)，智能推荐系统已实现完善但存在以下待修复问题：
- [ ] **[严重] 前端行为追踪链路未接通**: `recordBehavior`、`markClicked`、`markPurchased` mutation 在前端已定义但未被任何组件调用（`apps/client/app/(tabs)/index.tsx` 商品曝光/点击、`apps/client/app/product/[id].tsx` 详情页浏览/加购、`apps/client/app/components/product/RelatedProducts.tsx` 推荐点击/购买均需接入）
- [ ] **[严重] 首页推荐硬编码 userId=1**: `apps/client/containers/home/HomeContainer.tsx:61` 写死 `userId=1`，所有用户看到相同的推荐结果，应改为从 auth context 获取当前登录用户ID
- [ ] **[中] 测试覆盖严重不足**: 仅1个测试文件10个测试方法（仅覆盖 `RecommendationDomainServiceImpl`），缺少5种算法单元测试、应用层测试、GraphQL DataFetcher 测试、仓储层集成测试、控制器集成测试
- [ ] **[中] 缺少 Dockerfile**: 与 `user-service`、`product-service`、`order-service`、`payment-service` 等同类服务相比，推荐服务无法容器化部署
- [ ] **[中] 缺少离线计算/定时任务**: 流行度推荐依赖预计算结果但无 `@Scheduled` 自动计算；商品相似度矩阵需手动触发更新，无自动定时更新机制
- [ ] **[低] 后台管理功能冗余**: `src/views/recommendation/index.vue`（规则类型 BOOST/FILTER/RE_RANK/EXCLUDE）与 `src/views/sms/recommendation/index.vue`（规则类型 COLLABORATIVE/CONTENT_BASED/HYBRID/POPULARITY）两个推荐规则管理入口功能重叠，建议统一合并
- [ ] **[低] 后台缺少推荐效果数据看板**: 后端 `/api/admin/recommendation/statistics` 已提供统计数据，但后台管理缺少对应数据展示页面（CTR趋势图、转化率、推荐覆盖率等）

### [Pending Features]

- [ ] 添加 GraphQL Federation 统一查询层 (整合多个微服务的GraphQL端点，提供统一API)
   - [ ] 需要端到端运行时验证：启动所有子图服务 + gateway，用 GraphQL 客户端请求跨子图联合查询（如通过 cart 查 product），验证 Federation 路由和实体解析正确

- [] [关于ERC/MES方面的汇总改进意见](./TODO_ERP.md) 按照这个进行完善

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)
