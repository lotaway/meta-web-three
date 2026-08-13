# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

- [ ] 当前 `/developer/register` 接口存在安全隐患：
  - 任何人都可以无限制调用
  - 没有 CAPTCHA、没有邮箱验证、没有 IP 限流
  - 唯一的防护是注册后需要管理员人工审批（PENDING → APPROVED）
  对于 B2B 场景，更安全的做法是：
  1. __不暴露公开注册 API__，改为通过商务渠道线下收集开发者信息，由管理员在 backstage-admin 中创建账号
  2. 如果必须保留公开注册，至少加上 __CAPTCHA + 邮箱验证 + IP 限流__ 三层防护

## [区块重组处理](./TODO_BLOCK_REORG.md)

# 待决议功能

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

- [x] AI辅助购物（以图搜图、智能匹配、文本纠错、商品推荐）
  - [x] 文本纠错（LLM 优先，本地词典兜底）
  - [x] 智能匹配（向量检索 Milvus / 内存兜底）
  - [x] 以图搜图（multipart 上传 + 图像向量检索）
  - [x] 后台配置（AI Provider / Milvus / 索引重建 / 日志）
  - [ ] 商品推荐（沿用已有 recommendation-service 实现，未改动）
