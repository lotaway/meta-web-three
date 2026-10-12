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

> 方案与映射表见 [docs/microservice-consolidation.md](./docs/microservice-consolidation.md)。
> 原则：每完成一个合并立即「构建 + 单服务启动 + 冒烟」，通过后再进行下一个。

### Phase 0 — 基线与映射


### Phase 1 — ERP 领域（7 → 2）


### Phase 2 — 供应链（8 → 2）


### Phase 3 — 电商 mall（13 → 4）


### Phase 4 — 平台（9 → 2）


### Phase 5 — 工厂 / AI / 区块链（8 → 3）

  - 说明：mes 另有 6 个未实现仓储（`AndonTypeRepository`、`AndonLevelRepository`、`EquipmentStatusConfigRepository`、`EquipmentStatusTransitionRepository`、`EquipmentTypeRepository`、`WorkOrderSplitRuleRepository`），其消费方均无人引用，本次不做；将来需要时按 `InMemory*` 模式补。背景：`79cbe515` 仅为提供 `DomainEventPublisher` 加了 `@ComponentScan("com.metawebthree.common.event")`，mes 领域 bean 从未注册，本次显式扫描才暴露这些半成品

### Phase 6 — 收尾

- [ ] k8s 清单同步——**暂缓（需单独立项）**：`k8s/` 为合并前遗留脚手架（`deploy-all.yaml` + `services/extended-domain-services.yaml` 被 `deploy.sh` 应用，其余 `services/*.yaml`、PV、autoscaling、ingress、`digital-twin/*`、docs 均为旧服务名，且无集群可校验）。整体改写风险高、不可回归验证，故本轮不改；保留此条为后续独立任务。
  - 第 1 次回归：构建 BUILD SUCCESS，但 `platform-service` 启动失败——Dubbo triple 协议 `dubbo.protocol.port: -1`（自动）在 14 服务并发启动时争抢 50051/50052，`Address already in use`
  - 修复：`run-server.sh` 按 `服务 HTTP 端口 + 10000`（与 gateway 20881 约定一致）为每个服务注入唯一 `-Ddubbo.protocol.port`，消除并发绑定竞争（20081–20125 段确认空闲）
  - 第 2 次回归：BUILD SUCCESS，**14/14 服务 STARTED**（gateway 69s / platform 79s / factory 107s，全部存活 40s+）
