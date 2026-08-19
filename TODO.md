# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

- [x] `/developer/register` 接口已加上三层防护：`/developer/captcha/generate` 图形验证码、`/developer/email/send-verification-code` 邮箱验证码、按 IP 的 Redis 限流（另叠加 resilience4j `developerRegister` 限流），注册后仍保留管理员人工审批（PENDING → APPROVED）

## [区块重组处理](./TODO_BLOCK_REORG.md)

# 待决议功能

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)
