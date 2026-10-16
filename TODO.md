# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

## [区块重组处理](./TODO_BLOCK_REORG.md)

## 待决议功能

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)
- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)
- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

## 依赖安全（Dependabot）

GitHub Dependabot 告警共 53 条（High 20 / Medium 25 / Low 8）。后端 Maven 补丁已在本次处理（postgresql、minio、graphql-java、rhino、clickhouse-jdbc 版本升级）。前端 npm 依赖告警已于本次全部处理完毕：

- **electron**（37 条）：`apps/digital-twin/system-management` `"33"` → `41.10.7`（覆盖全部受影响范围 `<41.10.6`）。`npx vite build` 通过，vite-plugin-electron/renderer、plugin-react 与 electron 41 兼容。⚠️ 桌面端自动更新/沙箱等运行时回归仍需在桌面环境人工验证。
- **tinymce**（5 条）：`apps/backstage-admin` `^6.8.6` → `tinymce@7.9.3` + `@tinymce/tinymce-vue@6.3.0`；改为 npm+Vite ES6 模块打包（side-effect 导入皮肤/插件），删除 `public/tinymce6.8.6`，新增 `Tinymce/upload.ts` 与 `src/types/tinymce.d.ts`；已隔离构建并用 Playwright 无头冒烟验证（皮肤/中文 i18n/iframe 内容/无 console 报错）。
- **vite**（3 条）：`apps/digital-twin/system-management` `^5.0.8` → `^6.4.4`（受影响范围 `<=6.4.2`）；`@vitejs/plugin-react` `^4.2.1` → `^4.7.0`；`npm dedupe` 移除 vitest 嵌套的 vite 8(rolldown)；`npx vite build` 通过。`apps/backstage-admin` 已在 vite `^7.3.6`，不在受影响的 `<=6.4.2` 范围，无需变更。
- **turbo**（2 条）：`apps/client/turbo-module/appsdk` `^1.10.7` → `^2.9.14`（实装 2.11.8）；`turbo.json` `pipeline` → `tasks`，并补充 `packageManager: yarn@1.22.22`（turbo 2 必需）；`npx turbo run build:android --dry` 解析正常。

### 升级过程中暴露的既有问题（非本次引入，需另行排期）

- [ ] **backstage-admin 构建损坏**：`npm run type-check` / `build` 无法通过（约 770 个既有 TS 错误；`src/views/crm/index.vue` 等模板缺少结束标签）。因此 tinymce 改动采用隔离工程 + 无头冒烟验证。
- [ ] **digital-twin 仓库组件测试陈旧**：`AlertFlow.test.tsx`(20 失败) 与 `RealTimeDataDisplay.test.tsx`(15 失败) 共 35 条断言与组件实现不匹配（`共 N 条` / `暂无告警` / `role=status` / `data-testid=mock-line-chart` 等）。测试在组件重构后未同步，此前被 vitest/vite 版本错配导致的“整文件 JSX 解析失败”掩盖。已从 vitest 排除 `src/renderer/e2e/**`（Playwright 用例，不应由 vitest 收集）。
- [x] **digital-twin typecheck 被 native 产物阻塞**：已修复 `system-support/src/tts.rs` 两处编译错误（`TTSVoice` 补 `serde` 派生、`synthesize_speech` 末尾返回 `Ok(())`），`npm run build:native` 成功生成 `native/index.node`；`npm run typecheck` 与 `npm run build` 均通过（EXIT=0）。
- [ ] **`apps/client/.gitignore` 忽略 `*.lock`**：`yarn.lock`（含 RN 与 appsdk 依赖精确版本）未纳入版本库，依赖不可复现，Dependabot 仅能按 `package.json` 版本范围告警。

## 运行环境问题（logs/run-server 近期日志）

已于本次定位根因并完成代码/配置修复；剩余一步需在能连通 Zookeeper / PostgreSQL / 外网的环境做回归验证。

已修复内容（根因与改动）：

- **gateway**：根因并非“hosts 缺失”，而是各服务以机器 FQDN（`GUNDAM.mshome.net`）注册到 Zookeeper，网关 `OpenApiConfig.cacheApiDocs` 按注册地址拉取 `/v3/api-docs` 时 DNS 解析失败（Netty NXDOMAIN）。修复：`server/common/.../application-common.yml` 为所有服务开启 `spring.cloud.zookeeper.discovery.prefer-ip-address: true`，改为按 IP 注册。
- **factory-service**：实际不是“连接失败”（连接与认证均成功），而是缺少 `devices` / `production_lines` 关系（`relation "devices" does not exist`，合计 1400+ 次）。根因：数字孪生服务并入 factory-service 时遗漏了其表结构。修复：把 `workshops` / `production_lines` / `devices` / `alerts` / `alert_rules` 补入 `db/schema.sql`，并新增幂等增量脚本 `db/migration/V20261016__digital_twin_tables.sql`。
- **mall-trade-service**：根因是沙箱无外网出口，binance / coinbase / okx 全部 Connect timed out，而 `PriceEngineServiceImpl` 每 5 秒重试并刷 ERROR。修复：新增 `PriceEngineProperties`，价格引擎加熔断降级（连续失败即暂停外呼并仅告警一次）、可配置 `mock-prices` 兜底、把 `PriceEngineServiceImpl` 逐次 ERROR 与 `ExternalPriceServiceImpl` 每源失败日志降为 debug。另修复 `payment/config/HttpClientConfig` 中自定义的 `new ObjectMapper()` Bean（缺少 `JavaTimeModule`，导致 `/prices/{symbol}` 等返回 `LocalDateTime` 的接口序列化 500），改为复用 Spring Boot 自动装配的 ObjectMapper；`mall-trade-service` 已 `mvn clean package` 通过。

- [x] **运行环境回归验证**（已于 2026-10-16 在本地环境完成，三服务共用 Zookeeper / PostgreSQL）：
  - gateway：`prefer-ip-address` 生效后各服务按 IP 注册（`172.28.192.1:18084/18101/19101`）；网关 api-docs 缓存日志 `Caching API docs for 3 services` → `Fetched 134 schemas from service factory-service` / `Fetched 119 schemas from service mall-trade-service`，`GUNDAM.mshome.net` NXDOMAIN 计数为 0。
  - factory-service：已在 `192.168.1.194:5432/metawebthree` 应用 `db/migration/V20261016__digital_twin_tables.sql`（5 表 + 11 索引，幂等），运行期无 `relation/column does not exist`，`workshops` 插入成功，`devices` / `production_lines` 查询正常。
  - mall-trade-service：熔断降级后无逐次 ERROR，仅一条 `All price sources unreachable; pausing external price refresh for 60s`；配置 `payment.price-engine.mock-prices` 后 `GET /prices/BTC-USD` 返回 200（`price=50000`、`source=mock`、`timestamp` 正常序列化）。

## 微服务收敛（46 → 15）

### 工厂 / AI / 区块链（8 → 3）

- [x] MES PostgreSQL / MyBatis-Plus 持久化与后端入口：安灯类型、等级、事件、设备类型、状态配置、状态转换及工单拆分规则；已通过 7 项拆单测试和 8 项真实 PostgreSQL/接口测试，不以内存实现作为完成标准。
- [ ] 目标环境部署验收：备份并应用 MES 增量脚本，核对存量表结构和数据约束，验证完整 Spring Boot 启动及网关接口冒烟；步骤与当前边界见 [server/README.md](./server/README.md#mes-持久化与安全拆单factory-service)。

### MES 工单拆分的后续优化

- [ ] `BY_BOM`：读取真实 BOM，明确版本、生效时间、单位换算、损耗率与子装配数量关系；区分物料需求展开和需要下达的子装配工单，保留物料/工单溯源。原实现未读取 BOM，仅按数量切单，不应继续执行。
- [ ] `BY_PROCESS`：基于真实有效工艺路线生成工序任务，明确前后依赖、并行/返工、派工及完工汇总规则。工序数不是产量分母，100 件经过三个工序不能按 33/33/34 件拆成三个生产工单；移除写死三个工序的假实现。
- 当前先支持 `BY_QUANTITY` 批量拆分与 `MANUAL` 手动数量拆分：仅拆分已保存的草稿工单，数量为正且总和守恒，父子关联、事务回滚、防重复拆分；保留父工单原始数量。`BY_BOM` / `BY_PROCESS` 在业务规则完善并通过测试前明确拒绝执行。
- [ ] 子工单合并/完工汇总：明确父单汇总与子单执行的区别，不能重新启用父单生产同时保留可执行子单，不能覆盖已完成子单状态；当前不开放旧合并逻辑。
- [ ] 设备条件表达式、事件动作及超时自动转换：需定义执行器、事件来源和授权审计；当前手动转换不能静默忽略这些限制。

