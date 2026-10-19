# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

- [x] **网关鉴权默认 JWT 密钥过弱**（已修复并验证，2026-10-16）：`server/common/.../application-common.yml` 的 `jwt.secret: mwt123123` 仅 72 bit，而 jjwt 0.11.5 对 HS512 要求密钥 ≥ 512 bit，签发与验证均抛 `WeakKeyException`。修复：改为 `secret: ${JWT_SECRET:mwt-dev-only-jwt-secret-please-override-0123456789abcdefghijklmnopqrstuvwxyz}`（76 字符，满足 HS512），`JwtUtil` 的回退默认也由 `123123` 改为同样满足长度的值；保留 `JWT_SECRET` 环境变量注入，生产可覆盖。验证：三服务以默认配置启动，直接 `POST /admin/login` 取得令牌后经网关访问四个 MES 接口全部 200（无需再 `--jwt.secret` 覆盖）。
- [x] **`JwtUtil.generate` 丢失 subject**（已修复并验证，2026-10-16）：`JwtUtil.generate(subject, claimsMap, expiration)` 原为先 `setSubject(subject)` 再 `setClaims(claimsMap)`，而 jjwt 0.11.5 的 `setClaims` 会整体替换 claim 集，导致生成的令牌不含 `sub`；`UserJwtUtil.getUserId` 解析 subject 失败，使登录令牌一律无法通过校验。修复：改为先 `setClaims(claimsMap)` 再 `setSubject(subject)`（连同 `iat`/`exp`），并注释说明 `setClaims` 的替换语义。验证：`mall-user-service` 实际签发的登录令牌解码后为 `{"role":"ADMIN","name":"admin","sub":"1","iat":...,"exp":...}`，`sub` 已存在且能通过网关校验。
- [x] **网关鉴权排除路径配置未生效（YAML 缩进 + record 构造绑定双重缺陷）**（已修复并验证，2026-10-16）：`server/gateway/src/main/resources/application.yml` 中 `excluded-path-patterns`、`excluded-path-keywords`、`rbac-excluded-path-patterns` 原缩进为 2（挂在 `gateway:` 下），而唯一读取它们的 `GatewayAuthProperties` 前缀为 `gateway.auth`（需缩进 4），导致三个列表为空：所有路径（含 `/mall-user-service/admin/login`、`/actuator/**`、`/v3/api-docs`、`/ws/digital-twin`）均要求令牌，登录经网关返回 401 `AUTH_HEADER_INVALID`。修复一：把这三块移入 `gateway.auth` 之下。但修正缩进后经 `/actuator/env` 确认属性已进入 Environment（`gateway.auth.excluded-path-patterns[0..25]`），而 `/actuator/configprops` 仍显示 `excludedPathPatterns:[]`——根因是 `GatewayAuthProperties` 为 record 却额外声明了一个未使用的无参构造器（`public GatewayAuthProperties(){ this("Authorization",...); }`），Spring Boot 无法据此确定唯一构造器，构造绑定失效（记录各字段均为 null/空）。修复二：删除该无参构造器（全仓库无引用），使 record 仅剩规范的规范构造器。验证：`mvn -o -DskipTests -pl gateway -am clean package` 成功；默认配置启动 gateway(18101)+mall-user-service(10001)+factory-service(19101)，`configprops` 显示 `excludedPathPatterns` 等已正确绑定；`POST http://127.0.0.1:18101/mall-user-service/admin/login`（无令牌）→200 且返回 239 字符令牌；带该令牌访问 `/factory-service/api/mes/andon/types`→200；不带令牌访问同一受保护路径→401，排除项与鉴权均按预期工作。

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

- [x] **backstage-admin 构建损坏**（已修复并验证，2026-10-17）：`npm run type-check` 与 `npm run build` 均通过（EXIT=0；type-check 0 error，vite build 43.3s 成功）。实际既有错误为 408 条（非 TODO 所述约 770）。修复要点：（1）`element-plus@2.12` 的 `el-table-column` 作用域插槽 row 类型为 `DefaultRow = Record<PropertyKey, any>`，与具体 DTO 不兼容，导致 326 处 TS2345；在调用点插入 `as <DTO>` 断言（Element Plus 作用域插槽固有局限，逐点收敛而非全局放宽类型）。（2）6 处 locale 重复键（TS1117）：保留 `title` 为页面标题，字段标签改名 `titleLabel`，同步更新 `crm/opportunities`、`crm/tickets` 模板。（3）新建共享 `src/types/ui.d.ts`（`TagType`）并统一 el-tag `type` 助手；`rma/constants.js` 改为 `constants.ts`。（4）按 MyBatis-Plus `IPage` 约定把列表接口改为 `{ records, total }`（rma / dom / tenant）。（5）其余类型修复：`ai-warehouse` status 联合、`bi/*`、`dom`、`solana`、`commission`（API 双重包装 `CommonResult`）、`developer-portal`、`mes/scheduling`、`mes/scada` 图标名等。（6）`src/views/crm/index.vue` 缺少 `</script>` 结束标签，是 `vite build` 的首个致命错误。（7）删除 4 个孤立且损坏的死代码文件 `src/views/pms/payment/{index,reconciliation,refund,setting}.vue`（未被路由/引用，自 `02b3c4d8` 起所有引号被错误转义、引用代码库中不存在的 API，无法编译）。
- [ ] **digital-twin 仓库组件测试陈旧**：`AlertFlow.test.tsx`(20 失败) 与 `RealTimeDataDisplay.test.tsx`(15 失败) 共 35 条断言与组件实现不匹配（`共 N 条` / `暂无告警` / `role=status` / `data-testid=mock-line-chart` 等）。测试在组件重构后未同步，此前被 vitest/vite 版本错配导致的“整文件 JSX 解析失败”掩盖。已从 vitest 排除 `src/renderer/e2e/**`（Playwright 用例，不应由 vitest 收集）。
- [x] **digital-twin typecheck 被 native 产物阻塞**：已修复 `system-support/src/tts.rs` 两处编译错误（`TTSVoice` 补 `serde` 派生、`synthesize_speech` 末尾返回 `Ok(())`），`npm run build:native` 成功生成 `native/index.node`；`npm run typecheck` 与 `npm run build` 均通过（EXIT=0）。
- [x] **`apps/client/.gitignore` 取消忽略 `*.lock`**（已处理，2026-10-19）：修改 `apps/client/.gitignore`，允许 lock 文件提交以确保依赖版本一致性。
- [x] **factory-service 数字孪生演示种子非幂等**（已修复并验证，2026-10-17）：`DigitalTwinDevDataInitializer` 原仅以 `devices` 是否为空作为守卫，却会 `createWorkshop("WS-01")`、`createProductionLine("LINE-01", ...)` 等；当 `workshops` 已存在而 `devices` 为空时，种子在中途因 `workshops_workshop_code_key` 唯一冲突失败（MyBatis 以 ERROR 打印，异常随后被 catch 为 `warn`，不阻断启动）。修复：改为按各实体自然键存在性守卫——`findByWorkshopCode("WS-01")`、`findByLineCode("LINE-01")`、每台设备 `findByDeviceCode(code)`、每条告警按 `findByDeviceCode(deviceCode)` 判空，移除粗粒度守卫；创建产线后再设置目标产量。新增 `DigitalTwinDevDataInitializerTest`（4 例：全缺失→全建、全存在→零写入、workshop 存在而 devices 为空→不重建 WS-01、命令服务异常→不阻断启动），`mvn -o -pl factory-domain/factory-service -am -Dtest=DigitalTwinDevDataInitializerTest test` 通过（Tests run: 4, Failures: 0, Errors: 0）。目标库原缺陷场景（`workshops=1`(WS-01)、`devices=0`）启动验证：日志出现 `SELECT ... FROM workshops WHERE workshop_code=?` 守卫并跳过，`INSERT INTO workshops` 计数 0，无 `duplicate key`/`workshops_workshop_code_key` 错误，`LINE-01` 顺利创建。验证后已回滚演示数据（`lines`/`devices`/`alerts` 恢复 0，仅保留原有的 WS-01）。
- [x] **数字孪生事件发布失败会中止命令流程与演示种子**（已处理，2026-10-19）：事件发布通道已由 RabbitMQ 迁移至 Kafka（`DigitalTwinEventPublisher`，实际类名与描述中的 `DigitalTwinCommandApplicationService`/`DevelopmentDataSeeder` 不同）。修复两处：(1) `publishToKafka` 原仅捕获 `JsonProcessingException`，`kafkaTemplate.send` 同步异常会沿命令流程上抛导致 500，现改为捕获 `Exception` 并仅记录日志，事件发布失败不再中止命令；(2) `DigitalTwinDevDataInitializer.seedDigitalTwinDemoData` 原为整体单个 try/catch，任一步失败即中止全部种子，现改为车间/产线/设备/告警四步各自 `seedSafely` 隔离。已新增 `DigitalTwinEventPublisherTest`（3 用例）及 `DigitalTwinDevDataInitializerTest.continuesSeedingWhenEarlierStepFails`，factory-service 全量测试 23 passed / 0 failed（8 skipped 为 MesPostgresTest 环境跳过），BUILD SUCCESS。

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
- [x] 目标环境部署验收（已于 2026-10-16 在 `192.168.1.194:5432/metawebthree` 完成，步骤见 [server/README.md](./server/README.md#mes-持久化与安全拆单factory-service)）：
  - 备份：`pg_dump` 全库落盘（169 表 / 169 COPY 段，`-- PostgreSQL database dump complete`），文件 `backup-metawebthree-20261016-192012.pre-mes.sql`。
  - 应用：`psql -v ON_ERROR_STOP=1 -f V20261015__mes_configuration_and_safe_split.sql` 单事务提交（12 表 / 14 NOT NULL / 11 索引），exit 0；重复执行幂等。
  - 结构核对：12 张 `mes_*` 表均存在，主键、10 项唯一约束、3 个外键（含 `mes_equipment_status_transition` 的两个复合外键）、2 个 CHECK（`split_quantity > 0`、`max_child_orders BETWEEN 2 AND 1000`）与 10 个命名索引（含 3 个部分唯一索引）齐全；存量未被改动（`schema.sql` 的 26 张 `mes_*` 表与库一致，迁移表按 README 设计仅走增量脚本）。
  - 启动：factory-service `Started FactoryServiceApplication in 84.1s`，schema 相关错误（`does not exist` / `BadSqlGrammar`）计数为 0；仅存量 Kafka 告警及 `DigitalTwinDevDataInitializer` 演示种子重复插入（非 MES，见下）。
  - 网关冒烟：gateway `Started GatewayApplication in 55.0s`，`Fetched 134 schemas from service factory-service`；携带 Bearer 令牌经网关访问 `/factory-service/api/mes/andon/types|levels`、`/equipment-types`、`/work-order-splits/rules` 均 200；写路径 `POST /andon/types`、`POST /andon/levels` 200 并成功持久化（GET 可回读、`LocalDateTime` 正常序列化）、重复 `type_code` 返回 400 领域错误；测试数据已清理（12 表恢复 0 行）。
  - 边界：网关鉴权依赖 HS512 密钥，dev 默认 `jwt.secret=mwt123123` 仅 72 bit，被 jjwt 0.11.5 拒绝，故本次以命令行 `--jwt.secret=<85B>` 覆盖并用同密钥签发的令牌完成冒烟；该默认配置缺陷另列条目。

### MES 工单拆分的后续优化

- [ ] `BY_BOM`：读取真实 BOM，明确版本、生效时间、单位换算、损耗率与子装配数量关系；区分物料需求展开和需要下达的子装配工单，保留物料/工单溯源。原实现未读取 BOM，仅按数量切单，不应继续执行。
- [ ] `BY_PROCESS`：基于真实有效工艺路线生成工序任务，明确前后依赖、并行/返工、派工及完工汇总规则。工序数不是产量分母，100 件经过三个工序不能按 33/33/34 件拆成三个生产工单；移除写死三个工序的假实现。
- 当前先支持 `BY_QUANTITY` 批量拆分与 `MANUAL` 手动数量拆分：仅拆分已保存的草稿工单，数量为正且总和守恒，父子关联、事务回滚、防重复拆分；保留父工单原始数量。`BY_BOM` / `BY_PROCESS` 在业务规则完善并通过测试前明确拒绝执行。
- [ ] 子工单合并/完工汇总：明确父单汇总与子单执行的区别，不能重新启用父单生产同时保留可执行子单，不能覆盖已完成子单状态；当前不开放旧合并逻辑。
- [ ] 设备条件表达式、事件动作及超时自动转换：需定义执行器、事件来源和授权审计；当前手动转换不能静默忽略这些限制。

