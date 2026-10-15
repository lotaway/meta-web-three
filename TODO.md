# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

## [区块重组处理](./TODO_BLOCK_REORG.md)

## 待决议功能

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)
- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)
- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

## 微服务收敛（46 → 15）

### 工厂 / AI / 区块链（8 → 3）

  - 说明：mes 另有 6 个未实现仓储（`AndonTypeRepository`、`AndonLevelRepository`、`EquipmentStatusConfigRepository`、`EquipmentStatusTransitionRepository`、`EquipmentTypeRepository`、`WorkOrderSplitRuleRepository`），其消费方均无人引用，本次不做；将来需要时按 `InMemory*` 模式补。背景：`79cbe515` 仅为提供 `DomainEventPublisher` 加了 `@ComponentScan("com.metawebthree.common.event")`，mes 领域 bean 从未注册，本次显式扫描才暴露这些半成品

