# TODO

Guidelines: 
* Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). 
* All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.
* After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/) or [Customer Client](apps/client/)

---

### [Backend Admin Missing]

The following backend services have been created, but lack corresponding admin and operation pages. Each needs to be added:

- mall-domain (11 services, most missing admin pages)
- ai-domain (3 services: ai-warehouse, forecasting, risk-scorer)
- factory-domain / mes-service (production management admin)
- blockchain-domain (2 services)
- erp-domain (6 services: finance, HR, invoice, project, report, settlement)
- platform-domain (7 services: commission, customer service, data analysis, media, message, notification, user behavior)
- supply-chain-domain (6 services: inventory alert, inventory, logistics, procurement, supplier, warehouse)

---

### [Pending Features]

[ ]- 在[gateway](server/gateway/)里依旧使用了大量*Client.java文件里虚假实现的服务调用，再次明确按照[README.md](server/README.md)里的方式定义proto，接着使用[Makefile](Makefile)生成interface到[common](server/common/)里，让被调用的服务去实现这个interface，再让调用方也就是[gateway](server/gateway/)通过@DubboReference引入使用。

[ ]- 添加智能推荐系统 (AI推荐算法，基于用户行为个性化推荐商品)

[ ]- 实现多租户SaaS架构 (支持多个商户/租户隔离，每个租户独立配置和数据)

[ ]- 添加 GraphQL Federation 统一查询层 (整合多个微服务的GraphQL端点，提供统一API)
   - ✅ 6 个子图端点：product(10082)、order(10084)、user(10083)、inventory(10105)、recommendation(10104)、cart(10089)
   - ✅ Gateway FederationRouter 已更新：SUBGRAPH_URLS + ROOT_FIELD_OWNER 包含 cart
   - ✅ 编译验证：cart-service + gateway 编译通过
   - [ ] 需要端到端运行时验证：启动所有子图服务 + gateway，用 GraphQL 客户端请求跨子图联合查询（如通过 cart 查 product），验证 Federation 路由和实体解析正确
   - [ ] server/gateway/src/main/java/com/metawebthree/gateway/graphql/FederationRouter.java 当前对引用的微服务使用了url这种硬性依赖的形式，是否可以调整为类似@DubboRefence的方式或者服务名的方式引用+延迟懒加载，而不是这种写死了服务名+端口

[ ]- 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

[ ]- 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

[ ]- 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

[ ]- 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

[ ]- 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

[ ]- 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

[ ]- 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

[ ]- [mall-recommand](server/mall-domain/recommendation-service/)已经存在，为什么还添加[另外的服务](server/ai-domain/recommendation-service/)？