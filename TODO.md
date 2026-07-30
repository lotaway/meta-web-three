# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### [GitHub Issues]

- [ ] **[#1] Solana 商城模板集成**: Token, NFT, SFT 创建与管理；Token 作为商品销售；活动和佣金功能
  - [x] Token/NFT/SFT 创建和管理合约
  - [x] Token 作为商品销售的商城前端集成
  - [x] 活动与佣金功能
  - 链接: https://github.com/lotaway/meta-web-three/issues/1

### 安全

- [ ] 当前 `/developer/register` 接口存在安全隐患：
  - 任何人都可以无限制调用
  - 没有 CAPTCHA、没有邮箱验证、没有 IP 限流
  - 唯一的防护是注册后需要管理员人工审批（PENDING → APPROVED）
  对于 B2B 场景，更安全的做法是：
  1. __不暴露公开注册 API__，改为通过商务渠道线下收集开发者信息，由管理员在 backstage-admin 中创建账号
  2. 如果必须保留公开注册，至少加上 __CAPTCHA + 邮箱验证 + IP 限流__ 三层防护


# 待决议功能

- [ ] 实现多租户SaaS架构

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

