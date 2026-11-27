# TODO

## 永久性规范（持续约束）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES)

> 审计记录（2026-11-27）：按 CHECK_LIST.md 对原 19 个 `[x]` 项逐项核查。
> 通过核心验收的 5 项（原 11、14、15、16、17）已从本文件删除，其遗留改进转入第 5 节；
> 其余 14 项未通过，已改回 `[ ]` 并在每项后附问题与建议修改方式。

## 可执行任务

### 1. 需人工决策

- [ ] 清理 common 模块剩余的扫描遗留，统一到自动装配。
  - 问题（HIGH）：`server/data-pipeline/src/main/java/com/metaweb/datasource/pipeline/DataPipelineApplication.java:12-14` 仍保留 `@ComponentScan(basePackages={"com.metaweb.datasource.pipeline","com.metaweb.common"})`，包名错误且与 `@SpringBootApplication` 重复扫描。
  - 问题（HIGH）：`server/gateway/src/main/java/com/metawebthree/GatewayApplication.java:1,13` 无 `scanBasePackages`，全量扫描 common；`common/elasticsearch/ElasticsearchConfig.java:13-19`、`MQProducer.java:20`、`MQConsumer.java:21` 的条件被绕过；common 的 `AiTestController.java:19-20`、`OperationLogController.java:16-17`、`AlertController.java:12-13`、`LogQueryController.java:13-14`、`PerformanceMetricsController.java:13-14`、`TraceController.java:14-15` 被注册进网关，而 `UserAuthFilter` 只作用于路由、不覆盖这些本地映射，形成未鉴权接口面。
  - 问题（MEDIUM）：大量 `@Component/@Service` 成为孤儿（trace/audit/metrics/alert/字段加密/租户过滤等特性静默消失），如 `TenantContextFilter.java:17`、`RequestIdFilter.java:17`、`TraceAspect.java:14-15`、`OperationLogService.java:11`、`EncryptionService.java:23`；`mallcatalog/config/RestClientConfig.java:9-26` 与 `malltrade/config/RestClientConfig.java:9-26` 重复实现 common 版本。
  - 建议：删除 data-pipeline 的 `@ComponentScan`；网关限定扫描自有包；common 能力类去掉 stereotype、统一由带 `@ConditionalOnProperty` 的 `@AutoConfiguration` 注册并补上下文测试；删除服务本地重复的 `RestClientConfig`。

- [ ] 过滤并迁移非网关服务间调用，为 internal-token 全量开启铺路。
  - 问题（HIGH）：`server/common/src/main/resources/application-common.yml:100-104` 与 `docker-compose.server.yml:31`、`docker-compose.digital-twin.yml:30` 把 internal-token 密钥硬编码为 `mwt-internal-secret-2024`，任何人可伪造网关令牌。
  - 问题（HIGH）：`gateway/.../FederationRouter.java:33-39,344-359` 的 GraphQL 子图调用不带 `X-Internal-Token`，而白名单不含 `/graphql/**`，全量开启后所有联邦查询会 401。
  - 问题（HIGH）：`InternalTokenServletFilterAutoConfiguration.java:31` 用 `@ConditionalOnMissingBean` 未按 bean name 限定，任何其他 `FilterRegistrationBean` 会静默关闭网关校验。
  - 问题（HIGH）：`WarehouseServiceHttpClient.java:15-27,44-56`、`InventoryServiceHttpClient.java:13-25,36-39` 直连内部服务并吞异常返回 `null`/`9999.0`/`0` 等假值；`InventoryServiceMockClient.java:9-27`、`WarehouseServiceMockClient.java:10-41` 是 `@Service` 生产 mock，且开关 `dom.rpc.real-clients-enabled` 无任何配置定义，prod 下上下文无法启动。
  - 问题（MEDIUM）：`InventoryAlertNotificationListener.java:21-31,86-150` 的 URL 默认是空且未定义，POST 不带令牌并吞异常；仅 factory-service 启用了 filter；`platform-service/.../cs/config/WebSocketConfig.java:8-19` 与 `data-pipeline/.../DashboardWebSocketConfig.java:20-21` 的 WS 握手无鉴权；`docs/service-dependency-snapshot.md:43-57` 仍写着已禁用的 Feign/HTTP 直连。
  - 建议：删除默认密钥并要求显式注入；给 federation 调用注入令牌；按 bean name 限定条件；改用现有 Dubbo RPC 或网关+令牌；mock 移到 `src/test`；共享 WS 握手拦截器经自动配置注册；更新文档。

- [ ] 把 `apps/client` 与 `apps/backstage-admin` 的测试覆盖补齐。
  - 问题（HIGH，client）：`apps/client/jest.config.js:11-25` 覆盖率只收集 `app/**`，而现有测试均不 import `app/**`；阈值受 `process.env.COVERAGE==='true'` 控制，`package.json:7-10` 从不设置，等于未启用；`.github/workflows/frontend-quality-gate.yml:6-8,29` 只跑 admin，client 无 CI。
  - 问题（HIGH，client）：五维未覆盖——结构仅 3 个组件测试；响应式只测纯函数 `breakpoints.test.ts:3-17`；视觉仅手工单屏 `e2e/visual.spec.ts:21-33`；性能仅 hook 单测；可访问性仅 `ErrorBoundary.test.tsx:45` 一次断言，无 axe。
  - 说明：backstage-admin 部分通过（11 个 spec / 52 用例、真实 e2e 视觉基线 `login-win32.png`）。
  - 建议：client 覆盖率收集扩展到实际源码目录，CI 增加 client job 并强制阈值；补结构/响应式/视觉/性能/a11y 测试；合并重复的 `useInfiniteScroll.test.ts` 与 `__tests__/useInfiniteScroll.test.tsx`。

- [ ] 清理两个客户端里的代码原则违规项。
  - 问题（HIGH，client）：注释残留（`src/lib/api/baseUrl.ts:4-18,30-38`、`contexts/AuthContext.tsx:87,91,126` 等）；5 个文件超 500 行（`app/search.tsx` 611、`components/product/SKUSelector.tsx` 558、`app/orders/index.tsx` 546、`app/auth/login.tsx` 546、`app/checkout.tsx` 517）；`AuthContext.tsx:69-200` 多个函数 26-40 行；`tokenStorage.ts:7-18` 的 `getToken()` 既清理又返回；`components/product/SKUSelector.tsx` 未被使用且与 `src/components/product/SkuSelector.tsx` 重复；`baseUrl.ts:7,25,27`、`client.ts:230-232` 硬编码端口与阈值；`interceptors.ts:6-80` import 时修改单例、`initializeApiInterceptors()` 从未调用。
  - 问题（HIGH，admin）：`apps/backstage-admin/src` 364 个文件残留 2175 行注释；63 个非 locales 文件有 924 行中文硬编码（`views/normal/login/index.vue:35,44,101,111,116`、`router/guard.ts:43`、`views/ai-shopping/index.vue:203-326` 等），违反 DEVELOPMENT.md:102；9 个文件超 500 行（`views/oms/order/orderDetail.vue` 712 等）；`views/cash/account/index.vue:46-48`、`cash/transfer/index.vue:47`、`cash/reconciliation/index.vue:47`、`cash/forecast/index.vue:47`、`cash/plan/index.vue:66,71,76` 用 `ElMessage.info('not implemented yet')` 代替 Add/Edit/Detail 实现。
  - 建议：拆分超限文件与函数、删除死组件、清除解释性注释、用户可见文案统一走 i18n、实现或移除未实现动作、阈值与端口配置化、显式初始化拦截器。

- [ ] 补齐两个客户端的响应式与可访问性缺口。
  - 问题（HIGH，client）：约 92 个 TSX 文件中仅 15 处 `accessibility*` 标注；`app/auth/login.tsx:227,241-248,292-296`、`forgot-password.tsx:209,240-248,321-325` 的按钮与 `TextInput`（`login.tsx:277-285`）无角色/标签/状态；`constants/Colors.ts:14` primary 对白底对比度 3.46:1、`:23` textSecondary 3.08:1、`:15` secondary 2.92:1，均低于 WCAG AA 4.5:1；`hooks/useResponsive.ts:13-24` 从未在生产使用；无溢出/横竖屏测试。
  - 问题（MEDIUM，admin）：`views/layout/components/Navbar.vue:77`、`views/home/index.vue:210,217,224`、`views/cart/index.vue:275` 等 9 处 `<img>` 无 `alt`；a11y/视觉门禁只覆盖 `/login` 一个路由、`e2e/quality.spec.ts:8` 固定 1024 宽度；`views/rma/RmaDetailDialog.vue:74` 等 4 处硬编码 768px 未用共享断点 mixin。
  - 建议：补齐交互元素标签/角色/状态、把对比度调到 AA、扩展门禁到已登录页面与 640px、统一断点来源并加视口测试。

- [ ] 让 `apps/backstage-admin` 的 API 客户端与 DEVELOPMENT.md 第 3 节保持一致。
  - 问题（HIGH）：`src/utils/http.ts:17-67` 的 `SERVICE_PREFIX_MAP` 只覆盖 47 个前缀；对 `src/apis` 中 629 个 `url:` 静态重放，193 个（31%）请求拿不到服务前缀，例如 `src/apis/afterSale.ts:57-59` 的 `/api/after-sale/list`、`dataAnalysis.ts:67` 的 `/api/v1/analysis/sales/trend`、`rma.ts:74,82-144`，经网关会 404。
  - 建议：补全前缀映射或让模块显式声明 serviceId；添加测试断言每个 `src/apis/*.ts` 的 URL 都能解析出服务前缀。

- [ ] 让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐。
  - 问题（LOW）：`DEVELOPMENT.md:67`、`apps/client/.env.example:11-15` 称 `NEXT_PUBLIC_BACK_API_HOST` 仅用于生成/web 构建，但 `apps/client/src/lib/api/baseUrl.ts:40-50` 在 RN 运行时使用；`DEVELOPMENT.md:76-78` 要求全局安装 `@openapitools/openapi-generator-cli`，实际 `apps/tools/OpenApiToTS.js:107` 用 `npx`。
  - 建议：删除运行时回退或在文档/env 示例中明确其为遗留运行时变量；把前置条件改为 npx 方式并说明需 npm 网络。

- [ ] 统一本地工厂单档部署的端口、环境变量与 compose 口径。
  - 问题（MEDIUM）：`apps/backstage-admin/src/composables/useDigitalTwinEvents.ts:18-20` 的 `VITE_WS_PORT` 默认 `10081`，而 `src/stores/cs.ts:11-13` 默认旧端口 `10091`（`docs/microservice-consolidation.md:12` 中已废弃）；`VITE_WS_HOST/VITE_WS_PORT` 未在 `.env.example` 与 `env.d.ts` 登记。
  - 问题（MEDIUM）：`docker-compose.digital-twin.yml:86-87` 重复 `restart: unless-stopped`；`:88-93` InfluxDB 容器 healthcheck 探测 `localhost:10101/actuator/health`（factory-service 端口），应为 8086。
  - 问题（MEDIUM）：`.env.example:30-39` 缺 `KAFKA_PORT1/KAFKA_PORT2`（`docker-compose.env.yml:173-174` 使用）；`docker-compose.server.yml:25-26` 硬编码 `10081:10081`，不支持 `SERVER_PORT_OFFSET`。
  - 建议：统一端口变量并登记；修复 healthcheck 与重复键；补齐 env；决定是否让 compose 支持端口偏移。

### 2. 需环境 / 权限

- [ ] 重新启用 `apps/client` 的 e2e 套件并修复其配置。
  - 问题（BLOCKING）：`apps/client/e2e/app.spec.ts:30-327` 约 35 个用例依赖页面对象中的大量 `testID`，但生产代码全局仅有 3 个（`app/(tabs)/index.tsx:39,185,202`），套件实际无法通过；只有 `smoke.spec.ts`/`visual.spec.ts` 用到现有 ID。
  - 问题（HIGH）：`apps/client/e2e/mock-gateway.js:13-14` 读 `MOCK_GATEWAY_HOST/PORT`，而 `e2e/README.md:19` 文档用 `HOST/PORT`，默认绑定 `127.0.0.1` 模拟器不可达；无 CI 执行；`e2e/visual.spec.ts:15` 声称设备矩阵，但 `detox.config.js:38-47` 只有一个 Android 配置，`visual-baselines` 只有 `home.png`。
  - 建议：补齐稳定 testID 或重写套件并跑通全量、提交通过日志；修正环境变量名并说明 `adb reverse`；增加 Detox CI 或修正"设备矩阵"表述。

### 3. 需决策 + 环境

- [ ] 修复 `apps/client` 的鉴权 token 存储。
  - 问题（BLOCKING）：`apps/client/src/lib/auth/tokenStorage.ts:3,23` 用 AsyncStorage 明文存储 Bearer token，无 `expo-secure-store`/Keychain/Keystore；应用含 `react-native-web`，web 端可直接被 XSS 读取。
  - 问题（HIGH）：`:5` 硬编码 24h TTL；`:7-18` `getToken()` 同时清理与返回；生成客户端 `api/generated.ts:28-30` 用 `credentials:'include'` 却从不附加 token，`src/generated/api/apis/UserControllerApi.ts:335,374` 要求显式 `authorization`，而 `contexts/AuthContext.tsx:93,127,253` 调用时未传；`src/lib/api/interceptors.ts:6-80` 靠 import 副作用注册，`initializeApiInterceptors()` 从未调用。
  - 建议：改用 SecureStore 或 httpOnly cookie；拆分读取与过期清理、TTL 配置化；为生成客户端加令牌中间件并显式初始化。

- [ ] 落地 FRONTEND_PRICEPLES 的视觉与五维质量门禁。
  - 问题（HIGH）：唯一可执行的五维门禁在 admin（`apps/backstage-admin/quality.config.ts:1-36`、`tests/quality-gate.spec.ts`、`e2e/quality.spec.ts`），且 CI `frontend-quality-gate.yml:6-8` 路径过滤仅 `apps/backstage-admin/**`；`apps/client` 没有 quality config、没有 test:quality、没有 CI job，仅 `e2e/visual.spec.ts:21-33` 单屏手工视觉检查，阈值 `0.002` 硬编码。
  - 建议：为 client 增加同口径的 `quality.config.ts` 与 `test:quality` 脚本，在工作流中加入 client job；阈值统一从配置读取；补齐或修正设备矩阵。

- [ ] 撤销已泄露的 classic PAT 并从 git 历史中清除。
  - 问题（HIGH）：泄露的 PAT 仍可从对象库取回（`git show 3c07b882:package.json` 仍含完整 `ghp_...`），该提交仍在 master；历史上有"确认 revocation 401，但用户决定不重写历史、接受残留风险"的记录，当前 TODO 已把该说明删掉，呈现为完全完成。
  - 建议：如已授权则 `git filter-repo`/BFG 重写并协调强推、用失败模式 gitleaks 全历史校验；否则把该项改为"已撤销、历史保留（接受风险）"并在当前文档记录，撤销证据单独留档且不得回写 token。

- [ ] 验证 zk + gateway + 数字孪生专用账号 token 的整条链路。
  - 问题（HIGH）："数字孪生专用账号 token" 不存在——全库只在本 TODO 与测试夹具 `UserAuthFilterWebSocketTest.java:37` 出现；admin 实际用普通登录 JWT（`apps/backstage-admin/src/composables/useDigitalTwinEvents.ts:51-53`），WS 拦截器不校验账号/audience。
  - 问题（HIGH）：整条链路无端到端验证，现有测试全是 mock/手工构造过滤器（`UserAuthFilterWebSocketTest.java:35-39`、`GatewayOnlyAccessIntegrationTest.java:36-58`、`DigitalTwinAuthHandshakeInterceptorTest.java:22-24`），没有真实 ZK 发现与 `lb:ws://factory-service` 路由；信任锚仍是硬编码默认密钥（`application-common.yml:100-104`）。
  - 问题（MEDIUM）：`FaultTolerantZookeeperDiscoveryClient.java:25-40` 吞掉 ZK 失败并返回空列表，链路降级不可见。
  - 建议：实现并配置专用账号与作用域校验（或改写 TODO 描述）；增加 ZK+gateway+factory-service 的 Testcontainers/compose 真链路冒烟；删除默认密钥并强制注入；ZK 失败改为可观测的领域事件。

- [ ] 补齐服务端事件 payload 并让前端无需回查。
  - 问题（HIGH）：Kafka 发布端 `DigitalTwinEventPublisher.java:23,56-189` 使用前缀 `digital-twin.*`，消费端 `DigitalTwinKafkaConsumer.java:50,60,70,...` 监听无前缀 topic，14 类事件全部无法送达。
  - 问题（HIGH）：即使送达，消费端 `:198-307` 广播的是原始 JSON 字符串，`apps/backstage-admin/src/composables/useDigitalTwinEvents.ts:40` 因 `event.data.eventId` 不存在直接丢弃；发布端 `:50-190` 用裸 HashMap、未调用 `DigitalTwinEventPayload.of()`，`ALERT_CREATED`/`PRODUCTION_OUTPUT_UPDATED` 缺 eventId 及 UI 所需 9 个字段。
  - 问题（HIGH）：`shared/event-sdk` 的 `KafkaEventConsumer.java:80,36,89-91` 把所有消息反序列化为 `BaseEvent`、丢弃子类字段并无检查强转；65 种事件仅 5 个 payload 类，且 sdk 无测试目录。
  - 问题（MEDIUM）：前端仅处理 2 类事件，`apps/backstage-admin/src/views/digital-twin/index.vue:38-176` 其余变更仍显式 GET 回查；`apps/client` 无事件消费者；事件名在服务端枚举、Kafka 字符串、前端常量三处重复。
  - 建议：统一 topic 常量、发布与消费共用 schema；解析并补全 payload；修复 event-sdk 类型解析并加往返测试；扩展前端 reducer 或收窄 TODO 声明；发布/消费失败走持久化 DLT 而非仅记日志。

### 4. 区块链 devnet 联调（新增，待测试币）

> 环境探测结论（2026-11-27）：Windows 原生缺 Solana/Anchor CLI 与 SBF 工具，必须在 WSL 中操作；
> WSL 已具备 solana-cli 3.1.10、anchor-cli 1.0.0、cargo-build-sbf 3.1.10（platform-tools v1.52），
> devnet RPC 与 npm registry 可达，`anchor build` 已成功产出 `.so` 与 IDL（项目在 `/mnt/c` 首次编译约 6 分钟）。
> 程序 ID 已同步为部署 keypair 的 `FiSvBZn8sSV3r8xC6PcEC59D8oBeDUj2sdS9Jdppq9oW`。

- [ ] 获取 devnet 测试 SOL：钱包 `3VgtrtoRzGPquNgt37RcQ8menVB3hGQTMxrLmSUx7yEE`。RPC `requestAirdrop` 已关闭（410）、旧 faucet 域名 TLS 不通、第三方 faucet 也走已关闭的 RPC，只能人工在 `https://faucet.solana.com` 完成 captcha 领取，或提供已充值的 devnet 钱包。
- [ ] 用部署 keypair 把 `solana_contract` 部署到 devnet，确认程序地址与后端 `blockchain.solana.program-id` 一致。
- [ ] 开启 `BLOCKCHAIN_SOLANA_ENABLED=true` 做链上冒烟：Token/NFT/SFT 创建与管理、商品上架/购买/下架、活动参与/领奖/Merkle 根、佣金关系与分佣、优惠券、treasury initialize/deposit/withdraw。
- [ ] 核对链上结果与 DB 写入、PDA 推导、Keccak Merkle 证明的一致性，记录差异与遗留问题。
- [ ] 联调结束后将开关恢复默认关闭，并更新本节状态。

### 5. 审计通过项的遗留改进（从原已完成项拆出）

- [ ] 校验鉴权流程的业务码：`apps/client/contexts/AuthContext.tsx:280,287-294` 未检查 `ApiResponse.data.code`，HTTP 200 但业务码非 `0000` 会被当作成功。
- [ ] 补齐 demo profile 自足性：`server/factory-domain/factory-service/src/main/resources/application.yml:85-100` 只有 dev/prod 的 config import，没有 `application-demo.yml`，`SPRING_PROFILES_ACTIVE=demo` 单独启动会缺数据源/ZK/token 配置。
- [ ] 处理休眠的 demo 种子 SQL：`.../db/data.sql:6-30` 插入 8 个 demo 设备/告警但无 `source` 列，一旦被指向会标记为 REAL；删除或补 `source='DEMO'` 并限定 demo/test。
- [ ] 真机数据自动注册设备：`ScadaDomainServiceImpl.java:89-97`、`TelemetrySourceIngestService.java:46-55` 只写 `TelemetryRecord`，prod 下设备只能手工注册；按遥测自动创建 `source=REAL` 的 Device。
- [ ] 修吞异常：`DigitalTwinDevDataInitializer.java:36-42` 捕获全部异常仅记录 message，改为分类异常并记录完整上下文。
- [ ] 领域层去框架依赖：`TelemetryMappingProperties.java:1,10` 位于 domain 却标 `@ConfigurationProperties`，移到 infrastructure config，domain 只保留值对象。
- [ ] 消除重复 bean：`TelemetrySourceIngestService.java:17` 的 `@Service` 与 `TelemetrySourceConfig.java:47-52` 的同名 `@Bean` 在启用来源时会触发 `BeanDefinitionOverrideException`；二选一并补上下文测试。
- [ ] 日志降噪与能力建模：`MqttTelemetrySubscriber.java:60` 逐条 INFO 记录报文；`MqttTelemetrySource.java:49-51` 的 `read()` 用抛异常代替实现；`TelemetryNormalizer.java:49` 空 catch 吞掉解析失败。
- [ ] InfluxDB 安全与可测：`InfluxTelemetryStore.java:48-51` 将 bucket/equipmentCode 直接拼入 Flux 字符串，存在注入风险；补写/查测试；`ScadaDomainServiceImpl.java:59-62,127-132` 通过领域端口依赖抽象而非具体类；去掉 `docker-compose.digital-twin.yml:77-81` 硬编码口令；拆分 `getDashboardMetrics`（`:145-192`）。
- [ ] 数字影子状态与死代码：`ScadaDomainServiceImpl.java:102-107` 只登记指令却立即 `markSent()`，状态机与"仅登记"文档矛盾；`MqttTelemetryService.java:18-21`/`MqttCommandPublisher.java:23-35` 的写回通路无人调用且开机建连；补边界测试。
