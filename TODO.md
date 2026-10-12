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
- [x] mes + digital-twin + production → factory-service（新模块，端口 10101；提交 7f62f53e、71ed7fc2）
  - [x] 脚手架：宿主 + 合并 548 个 java + 合并 schema/mapper 资源
  - [x] `PokaYokeService` 补 `@Service`（第一层缺失 bean）
  - [x] 新增 `AndonEventRepository` 内存实现（`InMemoryAndonEventRepository`，沿用 `InMemory*Repository` 模式）——被 `ScadaDomainServiceImpl` 使用
  - [x] 去掉 3 个无人引用的半成品服务的 `@Service`：`EquipmentStatusService`、`EquipmentTypeService`、`WorkOrderSplitServiceImpl`
  - [x] 解决 mes 控制器重复映射：`MesController`（`/api/mes`）的 equipment 段与 `EquipmentController`（`/api/mes/equipment`）冲突，删除前者 equipment 段
  - [x] 构建 `factory-service`（`mvn -pl factory-domain/factory-service clean install -Dmaven.test.skip=true`）
  - [x] 单独启动 + 校验端口 10101（STARTED）
  - [x] 冒烟：`/v3/api-docs` 200、`/api/mes/equipment` 200；`work-orders`/`production-tasks`/`digital-twin`/`alert-rules`/`v1/production` 500（dev 库缺表 `relation "..." does not exist`，属环境性）
  - [x] 删除旧模块 `mes-service` / `digital-twin-service` / `production-service`
  - [x] 同步旧模块引用：`docker-compose.digital-twin.yml`、`apps/digital-twin/README.md`
  - [x] 提交（英文 commit message）
  - 说明：mes 另有 6 个未实现仓储（`AndonTypeRepository`、`AndonLevelRepository`、`EquipmentStatusConfigRepository`、`EquipmentStatusTransitionRepository`、`EquipmentTypeRepository`、`WorkOrderSplitRuleRepository`），其消费方均无人引用，本次不做；将来需要时按 `InMemory*` 模式补。背景：`79cbe515` 仅为提供 `DomainEventPublisher` 加了 `@ComponentScan("com.metawebthree.common.event")`，mes 领域 bean 从未注册，本次显式扫描才暴露这些半成品
- [x] 更新 registry/端口映射（ai 10103 / blockchain 10114 / factory 10101）

### Phase 6 — 收尾

- [x] 网关路由/鉴权同步到 15 服务拓扑：`application.yml` 免鉴权/豁免路径、`GatewayAuthConfig` routeRoles、`CircuitBreakerFilter` 服务识别、`UserAuthFilterTest`（提交 a08c5b7d）
- [x] 前端服务名映射与端口：`apps/backstage-admin/src/utils/http.ts` SERVICE_PREFIX_MAP 改为合后服务名；`apps/digital-twin/system-management/{vite.config.ts,src/renderer/config/digital-twin.ts}` 默认端口 10102 → 10101；页面文案（提交 a08c5b7d）
- [x] 前端硬编码服务名同步：`apps/client/*`（promotion/order/cart/product/user/recommendation-service → 合后服务名）、`SERVICE_PREFIX_MAP` 补 `/api/forecasting`、`/api/v1/route-optimizer`（→ ai-service）与 `/project-service`（→ erp-management-service，控制器沿用原 base path）（提交 9b58ba7b）
- [x] `config/domain/*.yaml` 服务键同步为合后服务名并合并表清单（提交 a08c5b7d）
- [x] `server/README.md` wallet-service → blockchain-service；`docs/microservice-consolidation.md` Phase 5 状态更新为 ✅
- [x] 文档同步扩展：`server/README.md` 全面重写为合后 15 服务拓扑（目录结构、端口表、各服务职责/AI 辅助购物专节、微服务必填配置清单、支付模块路径）；`config/domain/README.md` 示例改为 `mall-trade-service`；`docs/microservice-consolidation.md` 目标名 `erp-finance-service` → `finance-service`
- [ ] k8s 清单同步——**暂缓（需单独立项）**：`k8s/` 为合并前遗留脚手架（`deploy-all.yaml` + `services/extended-domain-services.yaml` 被 `deploy.sh` 应用，其余 `services/*.yaml`、PV、autoscaling、ingress、`digital-twin/*`、docs 均为旧服务名，且无集群可校验）。整体改写风险高、不可回归验证，故本轮不改；保留此条为后续独立任务。
- [x] `run-server.sh` 全量回归
  - 第 1 次回归：构建 BUILD SUCCESS，但 `platform-service` 启动失败——Dubbo triple 协议 `dubbo.protocol.port: -1`（自动）在 14 服务并发启动时争抢 50051/50052，`Address already in use`
  - 修复：`run-server.sh` 按 `服务 HTTP 端口 + 10000`（与 gateway 20881 约定一致）为每个服务注入唯一 `-Ddubbo.protocol.port`，消除并发绑定竞争（20081–20125 段确认空闲）
  - 第 2 次回归：BUILD SUCCESS，**14/14 服务 STARTED**（gateway 69s / platform 79s / factory 107s，全部存活 40s+）
- [x] 修复 platform-service CS `UserQueryPortImpl` 默认 URL `http://user-service:10083` → `http://mall-user-service:10083`（随第 2 次回归构建验证通过）
- [x] `scripts/check-server-memory.sh` 复测内存：14 个 Spring 服务合计 **5.8 GB RSS / 0.8 GB 堆提交 / 7.0 GB 堆上限**（对比收敛前 46 服务数十 GB、21 个无法启动）
- [x] 全量改动提交（英文 commit message）
