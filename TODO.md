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

- [x] 盘点 46 个服务（模块、端口、应用类）
- [x] 目标映射表写入 docs/microservice-consolidation.md
- [ ] 全面梳理服务间调用关系（RPC/Feign/网关路由/前端 baseURL），形成合并前依赖快照
- [ ] 建立合并验证清单（构建 + verify-all.sh 启动 + 端点冒烟）

### Phase 1 — ERP 领域（7 → 2）

- [x] 试点：invoice-service → 并入 finance-service
- [x] 验证 finance-service 合并后单独启动 + invoice 端点冒烟（10111 端口释放）
- [x] reporting-service → 并入 finance-service
- [x] settlement-service → 并入 finance-service
- [x] finance-service（含 invoice+reporting+settlement）单独启动验证通过（57s）
- [x] hrm-service + crm-service + project-service → erp-management-service（新模块，端口 10125）
- [x] erp-management-service 单独启动验证通过（61s）；gateway crm 子图地址更新并重建验证
- [x] Phase 1 清理（pom/registry/Dockerfile/compose/firework/旧目录）
- [ ] Phase 1 全量回归（并入 Phase 6 统一执行）

### Phase 2 — 供应链（8 → 2）

- [x] inventory + warehouse + supplier + dom + inventory-alert → scm-inventory-service（新模块，端口 10105）
- [x] scm-inventory-service 单独启动验证通过（54s）；supplier Feign/dom RPC/网关 inventory 子图地址更新
- [x] logistics + procurement + rma → scm-logistics-service（新模块，端口 10107）
- [x] scm-logistics-service 单独启动验证通过（38s）；supplier Feign 指向 scm-logistics-service
- [x] 更新 registry/端口映射并逐个验证启动
- [ ] Phase 2 全量回归（并入 Phase 6 统一执行）

### Phase 3 — 电商 mall（13 → 4）

- [x] user + user-action + risk-control → mall-user-service（新模块，端口 10083）
- [x] mall-user-service 单独启动验证通过（51s）；网关 user 子图地址更新
- [x] order + payment + cart + promotion + after-sale → mall-trade-service（新模块，端口 10084）
- [x] mall-trade-service 单独启动验证通过（56s）；网关 order/cart 子图地址更新（/graphql/order、/graphql/cart）
- [x] product + mall-supplier + recommendation + review → mall-catalog-service（新模块，端口 10082）
- [x] mall-catalog-service 单独启动验证通过（52s）；网关 product/recommendation 子图地址更新（/graphql/product、/graphql/recommendation）
- [x] live + group-buying + social-commerce → mall-social-service（新模块，端口 10095）
- [x] mall-social-service 单独启动验证通过（43s）；live/group-buy 控制器可访问
- [x] 更新 registry/端口映射并逐个验证启动（10083/10084/10082/10095 冒烟通过）

### Phase 4 — 平台（9 → 2）

- [x] message + media + commission + cs + data-analysis + tenant + developer-portal → platform-service（新模块，端口 10085）
- [x] platform-service 单独启动验证通过（63s）；message/media/commission/cs/data-analysis/tenant/developer-portal 端点冒烟可达
- [x] 更新 registry/端口映射并验证启动（10085 冒烟通过）

### Phase 5 — 工厂 / AI / 区块链（8 → 3）

- [x] forecasting + route-optimizer → ai-service（新模块，端口 10103，提交 8387bc98）
- [x] ai-service 单独启动验证通过；forecasting / route-optimizer 端点冒烟可达（500 为 dev 库缺表）
- [x] wallet + traceability → blockchain-service（新模块，端口 10114，提交 6a29793e）
- [x] blockchain-service 单独启动验证通过（49s）；wallet / traceability 端点冒烟可达
- [ ] mes + digital-twin + production → factory-service（新模块，端口 10101；脚手架 + 构建通过，启动验证未通过）
  - [ ] **启动失败根因（待修复）**：mes 领域层有 4 个 `@Service` 依赖 7 个「只有接口、无实现、无数据表」的仓储，Spring 实例化时找不到 bean：
    - `ProductionTaskCommandService` → `PokaYokeService`（普通类缺 `@Service`；已补注解，属第一层原因）
    - `ScadaDomainServiceImpl`（被 `ScadaController`、`MqttTelemetryService` 使用）→ `AndonEventRepository`
    - `EquipmentStatusService`、`EquipmentTypeService`（均无人引用）→ `EquipmentStatusConfigRepository`、`EquipmentStatusTransitionRepository`、`EquipmentTypeRepository`
    - `WorkOrderSplitServiceImpl`（无人引用）→ `WorkOrderSplitRuleRepository`
  - [ ] 缺失仓储清单（无 DO / Mapper / 数据表）：`AndonTypeRepository`、`AndonLevelRepository`、`AndonEventRepository`、`EquipmentStatusConfigRepository`、`EquipmentStatusTransitionRepository`、`EquipmentTypeRepository`、`WorkOrderSplitRuleRepository`
  - [ ] 背景：commit `79cbe515`（修启动）为 `MesServiceApplication` 加 `@ComponentScan("com.metawebthree.common.event")` 以提供 `DomainEventPublisher`，mes 领域 bean 实际从未注册，服务等同空壳；本次合并显式扫描 mes 后这些半成品才暴露
  - [ ] 修复方案（二选一，另行处理）：① 最小修复——为 `AndonEventRepository` 补内存实现（沿用 `InMemory*Repository` 模式），并去掉无人使用的 3 个半成品服务的 `@Service`；② 完整补齐——为全部 7 个仓储实现内存版
  - [ ] 修复后：重建 + 单独启动 + 冒烟，再校验 Dockerfile/compose/端口接线并删除旧模块
  - [ ] 删除旧模块 `mes-service` / `digital-twin-service` / `production-service`（含 `server/factory-domain/digital-twin-service/run-dev.sh` 等引用）；同步 `docker-compose.digital-twin.yml`、`apps/digital-twin/README.md` —— 待 factory 启动验证通过后执行
- [x] 更新 registry/端口映射（ai 10103 / blockchain 10114 / factory 10101）
- [ ] factory 服务单独启动 + 冒烟（受阻于上方 mes 半成品问题）

### Phase 6 — 收尾

- [ ] run-server.sh 全量启动无回归
- [ ] scripts/check-server-memory.sh 复测内存，对比收敛前后
- [ ] 同步 server/README、部署文档、网关路由与前端 baseURL
- [ ] 全量改动提交（英文 commit message）
