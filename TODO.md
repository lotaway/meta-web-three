# TODO

## 永久性规范（持续约束）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES)

## 可执行任务

### 1. 需人工决策

- [ ] 清理 common 模块剩余的扫描遗留，统一到自动装配。
  - 问题：`server/common/src/main/resources/META-INF/spring.factories` 里 `EnableAutoConfiguration=...ElasticsearchConfig` 是 Boot 2 写法，Spring Boot 3.5.4 不再读取；`AutoConfiguration.imports` 里除 `ServiceGovernanceAutoConfiguration` 外还列了 `ServiceGovernanceProperties` / `ServiceGovernanceAspect` / `ServiceGovernanceController` 三个非 `@AutoConfiguration` 类；`MallCatalogApplication` 仍手扫 `com.metawebthree.common.elasticsearch` 与 `com.metawebthree.common.utils.RocketMQ`；各服务仍有逐条 `@Import(CommonSecurityConfig, DatabaseConfig, SwaggerConfig, WebConfig, SecurityInterceptor, ...)` 清单。
  - 2026-11-16 决策口径：安全相关默认全开；数据库、消息队列、缓存、检索等一切存储相关一律按属性开关，由实际服务显式决定；common 只负责统一依赖与统一配置，服务通过属性独立调整配置项。
  - 实施规格：默认开（仅按 web 类型 / bean 存在性条件装配，不设 enabled 开关）— `CommonSecurityConfig`、`SecurityInterceptor` + `WebConfig`、`PasswordConfig`、`PermissionAutoConfiguration`、governance 的 Resilience4j bean；按属性开（默认关，服务在自己 yml 显式开启）— `DatabaseConfig`、`RedisCacheConfig`、`ElasticsearchConfig`、RocketMQ 的 `MQProducer`/`MQConsumer`、Kafka 的 `DomainEventAutoConfiguration`（保留 `@ConditionalOnBean(KafkaTemplate)` 并补显式开关）。每个开关配 `@ConfigurationProperties`（统一前缀），服务可独立覆盖连接与池参数。`spring.factories` 只保留 `ApplicationContextInitializer=LocalConfig`；`AutoConfiguration.imports` 只列真正的 `@AutoConfiguration` 类；各服务移除 `@Import(common...)` 清单与 `scanBasePackages` 中的 common 包。
  - 2026-11-16 决策：`internal-token` 目标是禁止所有非网关调用，全量开启前须先完成下面「过滤并迁移非网关服务间调用」任务；`/governance/**`（含两个 POST 写端点）不是例外，适用同一调用原则——外部 / 客户端经 gateway 受 JWT 约束，服务之间通过 `@DubboReference` 调用治理 RPC、只做必要的用户 id 等上下文校验，不直连 HTTP 端点。现状治理只有 `@RestController`（7 个端点）、`protos/` 无治理服务、全仓无服务间调用治理端点的代码，因此本轮不新增治理 Dubbo 接口（无实际调用方，不做超前设计）；`CommonSecurityConfig` 的 permitAll 白名单继续不含 `/governance/**`，`internal-token` 全开后直连服务端口必须返回 401。
  - 关闭条件：公共 bean 只由自动装配 + 显式属性决定，`spring.factories` 不再含失效的自动装配段，各应用 `scanBasePackages` 只含自身包，编译通过且服务可启动。

- [ ] 过滤并迁移非网关服务间调用，为 internal-token 全量开启铺路。
  - 调用原则（2026-11-16 确定）：只允许两种方式——① 外部 / 客户端统一经 gateway 分流，受 JWT 约束；② 服务端之间用 `@DubboReference`（按 `server/README.md`「添加新服务清单」第 4 条），视作程序内部访问，只传必要的用户 id 等上下文，不经网关、不受 JWT 约束。治理端点（含 `/governance/**` 的 POST 写操作）同样适用：外部经 gateway，服务间走治理 RPC 且只做必要安全验证，不作为绕过网关的 HTTP 后门。禁止 `@FeignClient`；禁止写死 service url 或 HttpClient 的 ip:端口（违背原则）；外部系统（区块链 RPC / 支付 / 钉钉 webhook / LLM / MinIO）不属于服务间调用，保持现状。
  - 问题：`internal-token.enabled` 仅 `factory-service` 开启，其余服务注释称「仍有非网关内部调用」。按上述原则扫描出以下违规直连：
    - 服务→服务：
      - [x] `scm-inventory-service` → `scm-logistics-service`：原 `@FeignClient ProcurementServiceClient` 已改为 `@DubboReference SupplierOrderService`；`ProcurementService.proto` 新增独立 `SupplierOrderService`（`listSupplierOrders` / `getSupplierOrder`），scm-logistics 新增 `SupplierOrderRpcImpl` provider，两端开启 `@EnableDubbo`，移除 openfeign 依赖与 `@EnableFeignClients`。调用方签名未变。
      - [x] `platform-service` → `mall-user-service`：原 `UserQueryPortImpl` 的 RestTemplate + 写死 `mall-user-service:10083` 已改 `@DubboReference UserService.getUserProfile`；`UserService.proto` 新增 `getUserProfile` rpc 与 `UserInfoProto.nickname`，`UserRPCServiceImpl` 补 provider，删除 `user-service.url` 配置与 `UserInfoDTO` / `ApiResponse` 解析类。
      - [ ] `factory-service` → 三个 AI 服务（`AnomalyDetectionClient.java:20`、`ForecastingServiceClient.java:21,31`、`LocationRecommendationClient.java:22`、`AbstractAIClient.java:85` 直连 `localhost:8082/8083/8084`）；`ai-warehouse-service` → 下游 AI 端点（`ForecastingClient.java:17`、`RecommendationClient.java:17`、`RiskScoringClient.java:17`、`GenericAIClient.java:17`、`AbstractAIClient.java:61`）。
    - 网关自身直连下游（服务端之间，按原则也应 `@DubboReference`）：`gateway/client/RecommendationClient.java:22`（写死默认 `mall-catalog-service/api/recommendation`，15 个方法全部返回动态 `Map`，被 GraphQLDataProvider 用于联邦查询，`protos/` 无推荐服务定义——需先决定是否新建完整 `RecommendationService` proto + provider，本轮未改）；`gateway/auth/ApiKeyAuthFilter.java:53,159,235`（默认 `localhost:8080` 的 developer-portal）；`gateway/graphql/FederationRouter.java:351`（GraphQL 子图）；`gateway/config/OpenApiConfig.java:141-163`（文档聚合）。后两者是协议强制 HTTP（GraphQL federation / OpenAPI spec），能否纳入 Dubbo 需先定。
  - 建议：逐个调用点按「先补 proto + `make gen`，再用 `@DubboReference` 替换 HTTP 客户端，最后删除 URL 配置项与 HttpClient 依赖」的顺序改造；每改通一条就把对应下游服务的 `internal-token.enabled=true`，避免一次性全开打断未迁移调用。AI 类先确认它们是独立 Python 服务（8082/8083/8084 不在服务注册表、`protos/ai/` 只有 `RiskScorerService`/`UserRiskProfileService`，无 forecasting / anomaly / location 的 proto）还是已合并进 `ai-service`（10103，仅有 REST controller）；若确非 Java 微服务，需另定其合法性（归为外部系统或补 proto 纳入 Dubbo），不能默认沿用写死 URL。
  - 需决策：三个 AI 服务与 `ai-warehouse-service` 下游端点的归属（外部系统 / 纳入 Dubbo）；GraphQL 联邦与 OpenAPI 文档聚合的 HTTP 调用是否豁免；`ApiKeyAuthFilter` 对 developer-portal 的调用改 Dubbo 后是否需要新增 rpc。
  - 环境要求：需要各服务与网关同时启动的联调环境逐条验证；须另行获准。
  - 关闭条件：无 `@FeignClient`、无服务间写死 URL / ip:port 的 HttpClient；`UserAuthFilter` 未授权直连返回 401；每个下游服务 `internal-token.enabled=true` 且有真实链路验证记录。

- [ ] 把 `apps/client` 与 `apps/backstage-admin` 的测试覆盖补齐。
  - 问题：`backstage-admin` 只有 1 个测试文件（`tests/env.spec.ts`），却对应 93 个手写 `src/apis/` 模块；`apps/client` 有 9 个测试文件，全在 `lib` 层，没有组件 / 页面覆盖，且 e2e 套件被禁用。
  - 建议：为两个 app 的 API 模块与关键页面 / 屏幕补测试，让 e2e 用例能在 CI 运行，并断言行为而不是「没有抛异常」。
  - 需决策：先确定验收口径。DEVELOPMENT.md 只写了「add tests matching the project's test framework」；若要求数值化的覆盖率门槛，必须先定目标百分比。
  - 关闭条件：两个 app 都补上 API 模块与关键页面 / 屏幕的测试，断言行为而非 `not.toThrow()`，让 e2e 用例可在 CI 运行，并重跑各 app 的 typecheck 与测试。

- [ ] 清理两个客户端里的代码原则违规项。
  - 问题：DEVELOPMENT.md `:82-85` 要求代码文本一律英文（i18n 除外）、函数不超过 20 行、不吞异常、无隐藏共享状态、使用卫语句。静态扫描发现中文文本：backstage-admin 5751 行、client 891 行；catch 块 581 / 15 个，其中若干只记录日志就返回。
  - 建议：按 app 逐个处理：翻译注释、日志与错误文案（只保留 i18n 资源作为中文），把「记录后忽略」的 catch 改成显式错误状态或重新抛出，拆分超过 20 行的函数。每轮之后重跑该 app 的 typecheck 与测试。
  - 需决策：界定哪些中文字符串属于 i18n 资源（只有这些可以保留），批量改写前须先商定基于路径的分类规则。
  - 关闭条件：按 app 逐个把注释 / 日志 / 错误文案改为英文，把记录后忽略的 catch 改成显式错误状态或重新抛出，拆分超过 20 行的函数，每轮都通过 typecheck 与测试。

- [ ] 补齐两个客户端的响应式与可访问性缺口。
  - 问题：`@media` 只出现在 backstage-admin 的 4 个文件，client 为 0；`aria-*` 命中数 1 / 0；`apps/backstage-admin/src/styles/index.scss:32,37` 用 `outline: none` 移除了焦点轮廓；`apps/client` 没有无障碍属性，也没有 ErrorBoundary，并把方向锁死为竖屏（`app.config.ts:9`、`android/app/src/main/AndroidManifest.xml:19`）。
  - 建议：为两个 app 增加断点与弹性布局，恢复可见焦点态，补充语义标签 / `aria-*` 或 `accessibilityLabel` / `accessibilityRole`，并增加 ErrorBoundary 与加载失败时用户可感知的降级表现（FRONTEND_PRICEPLES 第 7、8 节）。
  - 需决策：提供设计基准 / 断点取值（仓库里没有设计稿）。若没有，默认采用通用断点集，并以可评审的变更提交。
  - 关闭条件：两个 app 都加上断点与弹性布局、恢复可见焦点态、补充语义标签 / `aria-*` / `accessibilityLabel` / `accessibilityRole`、加入带用户可见降级的 ErrorBoundary，并用各 app 的检查验证。

- [ ] 让 `apps/backstage-admin` 的 API 客户端与 DEVELOPMENT.md 第 3 节保持一致。
  - 问题：`package.json:18` 定义了 `generate:api`，`tools/generate-api.cjs` 调用共享的 `apps/tools/OpenApiToTS.js`，其默认输出为 `src/generated/api`，但该目录不存在；应用实际调用的是 `src/apis/` 下 93 个手写模块。
  - 建议：要么运行生成器并把调用方迁移到生成的客户端，要么在 DEVELOPMENT.md 与 app README 中记录这一有意偏离（包括为何保留手写模块、以及如何保持同步）。
  - 需决策：在建议的两条路线中二选一：运行生成器并迁移 93 个手写模块，或记录有意偏离。需与下面「对齐 DEVELOPMENT.md 第 3 节」一起决定，否则两处改动会冲突。
  - 环境要求（仅生成器路线）：后端 OpenAPI 文档可访问，且生成器工具链可用。
  - 关闭条件：要么迁移调用方并产出 `src/generated/api/`，要么把偏离说明落到 DEVELOPMENT.md 与 app README，并写明同步机制。

- [ ] 让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐。
  - 问题：`DEVELOPMENT.md:49-58` 要求 `.env.development` 定义 `NEXT_PUBLIC_BACK_API_HOST` / `NEXT_PUBLIC_BACK_API_DOC_HOST`，且两个 app 都输出到 `src/generated/api/`。实际上 `backstage-admin/.env.development` 只设置了 `VITE_BLOCKCHAIN_ENABLED=true`，其生成器读取 `.env,.env.local` 中的 `VITE_BASE_SERVER_URL` / `VITE_API_DOC_HOST`；`apps/client` 把 `NEXT_PUBLIC_*` 变量放在 `.env`，`apps/client/.gitignore:154` 忽略 `src/generated` 且 `api/generated.ts` 仍从 `@/src/generated/api` 导入（全新检出需先本地生成才能通过 typecheck）。`DEVELOPMENT.md:60` 只说明桌面端已迁到独立仓库 `lotaway/meta-note`，第 3 节只覆盖本仓库这两个 app。
  - 建议：选定一套 env 文件 / 变量约定，同步更新两个 app 与 `.env.example` 文件，并重写第 3 节描述各 app 的真实行为。
  - 需决策：在 `NEXT_PUBLIC_*` 与 `VITE_*`、`.env.development` 与 `.env` 之间选定唯一约定；需与上面「让 backstage-admin API 客户端与第 3 节一致」一起决定。
  - 关闭条件：更新两个 app 与 `.env.example` 文件，重写 DEVELOPMENT.md 第 3 节，描述各 app 的真实行为。

- [ ] 统一本地工厂单档部署的端口、环境变量与 compose 口径。
  - 问题：`docker-compose.env.yml` 的 kafka 用 `network_mode: host`，而 `docker-compose.digital-twin.yml` 注入 `SPRING_KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}`，容器里解析不到 `kafka:9092`；factory-service 只经 gateway `10081` 暴露（`ports: []`），但唯一的本地工厂端口表还没写进 README。桌面端 `WHISPER_URL` 与内置 `llama-server` 都用 `8080` 的冲突属于独立仓库 `meta-note` 的任务。
  - 建议：为 kafka 的 host 网络模式确定一个容器内可达的地址（或改 compose 网络口径）；把唯一端口表写进 README（gateway `10081`、factory-service `10101` 仅容器内、Whisper、llama-server、`5051` / `5050` / `8000`）。
  - 需决策：定唯一端口分配表与 compose 入口（确认以根 `docker-compose.yml` + `docker-compose.digital-twin.yml` 叠加为准）；env 命名规则需与「让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐」一并决定。
  - 关闭条件：只剩一份本地工厂口径，所有 `VITE_*` 在构建期生效且登记在 `.env.example`，容器起来后各页面经 gateway 连到正确地址，kafka 可解析。运行验证须另行获准。

### 2. 需环境 / 权限

- [ ] 重新启用 `apps/client` 的 e2e 套件并修复其配置。
  - 问题：代码层面已可加载，但 detox 套件从未在模拟器或真机上跑通，也没有 CI job 运行它；`e2e/setup.ts` 的 `E2E_SKIP` 开关、`e2e/detox.config.js` 的 Detox 20 配置与 `e2e/app.spec.ts` 的页面对象选择器都还没有真实运行记录。
  - 建议：至少运行一条真实用例（`E2E_SKIP` 不设置，CI 或有记录的手动真机检查均可）并记录结果，之后才能移除本项。不要重新引入全局跳过。
  - 环境要求：需要 Android SDK 与模拟器镜像（AVD 名通过 `DETOX_AVD_NAME` 指定，默认 `Pixel_7_API_34`），或一台已连接的 Android 真机（`-c android.attached.debug`，`DETOX_ADB_NAME` 指定设备名）。要长期保持套件可用，还需要具备同等运行时的 CI job，或一份有记录的手动真机检查。
  - 关闭条件：在 `E2E_SKIP` 不设置的情况下，于模拟器或真机上至少跑通一条真实用例并记录结果（必要时修正页面对象与真实界面不匹配的选择器）。

### 3. 需决策 + 环境

- [ ] 修复 `apps/client` 的鉴权 token 存储。
  - 问题：`src/lib/api/interceptors.ts:8,13,18` 通过 `localStorage` 读写 `auth_token`，而 React Native 运行时没有该 API（app 里也没找到 polyfill），且它不是安全存储；写进去的 token 既可能触发运行时 `ReferenceError`，又以明文落盘。
  - 建议：通过单一 token 存储模块使用 `expo-secure-store` / `SecureStore`（或平台 keystore），增加过期 / 刷新路径，并补一个测试确认目标运行时上该存储 API 可用。
  - 需决策：批准新增 `expo-secure-store` 原生依赖，并确定过期 / 刷新策略。
  - 环境要求：需要重新构建 Expo dev client（`expo run:android` 或 EAS dev build）。SecureStore 是原生模块，Expo Go 里无法验证，目标运行时可用性测试也需要这样的构建。
  - 关闭条件：所有 token 读写收敛到一个基于 SecureStore 的模块，加入过期 / 刷新处理，补一个断言目标运行时存储 API 可用的测试，并在构建好的 dev client 上验证。

- [ ] 替换 `apps/client` 里用抛异常代替实现的三个鉴权流程。
  - 问题：`contexts/AuthContext.tsx:237-238`、`:274-275`、`:285-286` 在手机号登录、验证码、忘记密码三处只有 TODO 注释加中文 `throw new Error(...)`，真实实现被注释掉。CHECK_RULE.md 禁止用抛异常代替实现。
  - 建议：把三个流程接到生成的 API 端点，错误消息保持英文，用单测覆盖成功 / 失败 / 限流场景，删掉无用的注释代码。
  - 需决策：提供三个端点的契约（手机号登录、验证码、忘记密码），并决定实现对接真实后端还是按契约实现。还要确定短信与限流测试允许的 stub 边界。
  - 环境要求：可访问的 auth API 或其 OpenAPI spec、验证码服务提供方，以及失败用例所需的限流 fixture。
  - 关闭条件：接通三个流程、错误消息保持英文、用单测覆盖成功 / 失败 / 限流、删除注释掉的死代码，并通过该 app 的 typecheck 与测试。

- [ ] 落地 FRONTEND_PRICEPLES 的视觉与五维质量门禁。
  - 问题：第 6 节（`:49-50`）要求可执行的视觉质量检测机制和量化差异标准；第 9 节（`:68-69`）要求自动化覆盖结构正确性、响应式布局、视觉一致性、性能指标与基础可访问性。`apps/backstage-admin` 与 `apps/client` 下没有任何视觉 / 截图 / a11y / 性能测试文件。
  - 建议：增加可运行的截图 diff / 视觉回归任务并给出明确的像素或感知阈值，把 a11y 与性能预算接入 CI；`backstage-admin` 用 Playwright，`apps/client` 用设备矩阵截图运行器。
  - 需决策：确定每个 app 的工具、视觉差异的数值阈值（像素或感知）、基线截图由谁批准，以及 a11y / 性能预算的具体数值。FRONTEND_PRICEPLES 第 6、9 节要求量化标准，但没有给出数值。
  - 环境要求：`backstage-admin` 需要 Playwright 浏览器二进制与基线图片，`apps/client` 需要设备矩阵截图运行器，还需要能存储并比对基线的 CI job。
  - 关闭条件：落地带明确阈值的可运行截图 diff 任务，把 a11y 与性能预算接入 CI，并给出一次通过、一次故意失败的运行作为证据。

- [ ] 撤销已泄露的 classic PAT 并从 git 历史中清除。
  - 问题：提交 `3c07b882`（2026-05-26）在 `package.json` 的 `repository.url` 里存了明文 `ghp_…` token；当前 `package.json`、`.git/config` 与 `remote.origin.url` 已清理，`lefthook.yml:23-28` 与 `.github/workflows/secret-scan.yml` 会阻断新增泄露，但该值仍可从对象库读出，且从未被撤销或清除。本机不存在 `~/.git-credentials`（`credential.helper=manager` 把凭据存在 Windows 凭据管理器里），所以仍在使用的 classic PAT 必须到 GitHub Settings 里核查并撤销，而不是删某个文件。
  - 建议：在 GitHub Settings 删除 / 撤销 classic PAT，换成细粒度 PAT 或操作系统凭据管理器，并用 `git filter-repo` / BFG 清除历史值，配合协同 force-push 与清除后的轮换。若不接受重写历史，需明确记录所接受的风险。
  - 需决策：批准（或拒绝）重写 git 历史。`filter-repo` / BFG 会改变每一个提交哈希，影响所有协作者、fork、未合并 PR 与 CI 引用，需要协同 force-push。若拒绝，替代方案是明确记录所接受的风险。
  - 环境要求：需要 GitHub 账号 Settings -> Developer settings 权限来撤销 classic PAT、仓库管理员权限执行 force-push、与所有协作者协调、以及清除后的凭据轮换。正在使用的凭据在 Windows 凭据管理器里，不在 `~/.git-credentials`。
  - 关闭条件：在 GitHub Settings 确认撤销，换成细粒度 PAT 或操作系统凭据管理器，通过协同 force-push 清除历史值（或记录所接受的风险），并重跑一次全历史密钥报告。

- [ ] 把性能流水线指向真实的 staging 目标。
  - 问题：`GET /repos/lotaway/meta-web-three/actions/variables` 返回 `total_count: 0`，因此 `vars.PERF_BASE_URL` 未设置，每次运行测的都是内置的 `127.0.0.1:18087` mock；从未触达真实服务。
  - 建议：把 `PERF_BASE_URL` 仓库 / 环境变量设为可达的 staging 网关，重跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL 且 `use_mock=false`。
  - 需决策：批准对 staging 做压测，提供目标 URL 与测试窗口，并决定每日定时任务是否也打它、还是只允许手动运行。
  - 环境要求：需要从 `ubuntu-24.04` runner 可达的 staging 网关、设置 `PERF_BASE_URL` 所需的仓库 / 环境变量写权限，以及 `workflow_dispatch` 权限。
  - 关闭条件：设置变量后跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL、`use_mock=false`，且没有 MOCK DATA 标注。

- [ ] 按「演示数据仅用于初始化／测试」的边界重建设备数据通路。
  - 问题：`DigitalTwinSimulator.java` 标 `@Profile("dev")` 且用 `@Scheduled(initialDelay = 5000, fixedDelay = 10000)` 随机改设备状态；`DigitalTwinDevDataInitializer.java` 标 `@Profile("dev")` 注册 8 台演示设备；`MqttConfig.java` 默认 `mqtt.enabled:false`。而 `docker-compose.digital-twin.yml` 设 `SPRING_PROFILES_ACTIVE=prod`，覆盖了 `application.yml:7` 的默认 `dev` —— 容器启动后既没有演示数据也没有采集数据，`/api/digital-twin/devices` 为空。演示／模拟数据只能出现在测试脚本或初始化演示，不能因此放弃真实数据读取与接入。
  - 建议：把模拟器与 8 台演示设备从 `dev` 抽到显式的 `demo` 初始化器（或移进 `src/test` 的 fixture），仅在用户选择「演示模式」时写入一次，不再定时改写运行时状态；`prod` 必须接到下面「协议无关的通用现场接入层」的真实数据源；MQTT 开关由 compose 显式注入并在 prod 打开。状态实体补 `source`（REAL / DEMO）/ `quality` / `timestamp`，前端在演示模式下明确标注数据非真实。
  - 需决策：演示模式的触发方式（env 开关 / 首次启动引导 / 设置页），以及演示数据与真实数据是否允许共存。
  - 环境要求：需要可达的 MQTT broker 与至少一个数据源（真实设备或 OPC UA / PLC 仿真器）才能验证 prod 通路；须另行获准。
  - 关闭条件：prod 能从真实数据源读到设备与遥测，`demo` 只做一次性初始化、不参与运行时状态变更，测试脚本里的模拟数据可复现，前端能区分 REAL / DEMO 与数据新鲜度。

- [ ] 建设协议无关的通用现场接入层（当前没有真实工厂，目标是后续最小修改）。
  - 问题：全仓没有 OPC UA / Modbus / 西门子 S7 等现场协议实现；`Device.mqttTopic` 只是实体字段；`MqttTelemetrySubscriber.java` 只订阅 `MqttConfig` 默认的裸前缀 `mes/equipment/`，没有 Sparkplug B、没有按 ISA-95 `Enterprise → Site → Area → Line → Cell → Device → Tag` 组织的 UNS、没有单位 / 时间戳 / quality 归一化，也没有断网缓存与补发。目前不存在可用的真实数据读取路径。
  - 选型约束：Apache PLC4X 1.0.0 用一套 `PlcDriverManager` API 覆盖 S7 / Modbus TCP / OPC UA，但官方 SPI / 驱动 / 工具以 Java 21 编译（仅 `plc4j-api` 保持 Java 17），运行时必须升到 JDK 21；否则只能锁 0.13.x（Java 11 基线）或改用 Eclipse Milo + Paho。MQTT 已有 `org.eclipse.paho.client.mqttv3:1.2.5`，Sparkplug B 可用 Eclipse Tahu。PLC4X 不负责 MQTT 发布，读到值后要自行编码交给 Paho。
  - 建议：先做协议无关的 SPI：定义 `TelemetrySource`（connect / read / subscribe / close）与配置化点位映射（连接串、tag 地址、单位、quality、采样周期），把现有 MQTT 订阅做成默认实现，再以可选 profile 挂载 PLC4X 驱动（S7 / Modbus / OPC UA）或 Milo；对外统一发布到按 ISA-95 层级组织的 UNS topic，边缘侧做死区过滤与断网缓存补发。换协议只改配置与一个适配器，业务代码不动。
  - 需决策：是否把 JDK 升到 21 以用 PLC4X 1.0.0，还是锁 0.13.x / 只做 MQTT + Milo；首批支持的协议集合；UNS 层级与 Sparkplug B 是否采用。
  - 环境要求：需要 OPC UA 服务器或 PLC 仿真器（如 PLC4X mock driver）、MQTT broker、OT 网络 VLAN 与证书。
  - 关闭条件：`TelemetrySource` 接口与配置化点位映射落地，至少两种来源（MQTT + 一种仿真 PLC/OPC UA）能把带 quality 与时间戳的点位送进 `factory-service` 并在 3D 页面显示，新增协议只需加一个适配器与配置。运行验证须另行获准。

- [ ] 实际接入 InfluxDB 时序库，不允许留空依赖。
  - 问题：`pom.xml:46-47` 声明 `com.influxdb:influxdb-client-java:7.2.0`，但 `src/main/java` 里零使用；`TelemetryRecord` / `ScadaTelemetryRecordDO` 走关系型 mapper，没有时序读写与历史回放；`ScadaController` 的 `/api/mes/scada/telemetry/{code}/range` 因此拿不到真正的时序数据，追溯与趋势面板缺数据底座。声明了依赖就必须实际接入，不接受直接删掉依赖了事。
  - 建议：新增 `InfluxTelemetryStore`（`InfluxDBClientFactory.create(url, token, org, bucket)` + `WriteApiBlocking` / `QueryApi`），在 `MqttTelemetryService` 的 `scadaDomainService.ingestTelemetry(...)` 路径与 `POST /api/mes/scada/telemetry/ingest` 写入 point（measurement=`telemetry`，tag=`equipmentCode`，field 为解析后的指标，timestamp 用采集时间）；把 `getTelemetryByRange` 改走 InfluxDB 范围查询，最新值查询仍可走关系库；连接参数（url / token / org / bucket）通过 `application-common*.yml` 注入并在 compose 起一个 InfluxDB 服务；写入失败要显式报错而不是静默吞掉。
  - 需决策：measurement / tag 命名与保留周期；InfluxDB 版本与是否启用鉴权 token；是否保留关系库中的遥测表做双写。
  - 环境要求：需要 InfluxDB 实例（目前 `docker-compose.digital-twin.yml` 只有 kafka / redis）与历史数据量评估。
  - 关闭条件：遥测可写入 InfluxDB 并按设备 / 时间范围查询返回，`/api/mes/scada/telemetry/{code}/range` 有真实数据来源，依赖不再是空声明。运行验证须另行获准。

- [ ] 明确「数字影子 / 闭环孪生」边界，并实现受控写回（如需）。
  - 问题：`ScadaController.java:58` 有 `POST /api/mes/scada/commands`，`MqttCommandPublisher.java` 能把指令发到 broker，但写回完全由人工触发，没有自动或受控闭环；按 Kritzinger 的分类，只有单向展示的是 digital shadow，而当前连展示数据都来自 `DigitalTwinSimulator`。README 却一直按「数字孪生」描述。这条边界要在真实数据源接通后再定。
  - 建议：若需要闭环，做「建议 → 人工确认 → 白名单点位写回 → 审计」链路，控制逻辑与安全联锁仍留在 PLC；若不需要，把产品命名与 README 统一为「数字影子 / 可视化监控」，不提供写回。
  - 需决策：是否允许写回产线、审批流与白名单点位范围、产品命名是否调整。
  - 环境要求：需要可写测试点位、PLC 侧安全联锁与写操作审计存储。
  - 关闭条件：闭环链路可用且有审计记录，或文档明确单向边界并统一命名。

- [ ] 验证 zk + gateway + 数字孪生专用账号 token 的整条链路。
  - 问题：缺少真实运行记录。待验证：zk 注册、`discovery.locator` 的 `/factory-service` 前缀剥离、显式 `lb:ws://factory-service` 路由、WS 握手 `?token=` 校验与转发前剥离、5 秒内部令牌窗口、login / refresh / 401 重试；当前都只有静态核对与单测代码。
  - 建议：获准后做一次端到端：未带 token 的 REST / WS 均 401，带有效 JWT 能拿到设备列表与 5 类事件；再验证账号过期前 refresh 成功、被注销后 401 触发重登，并记录结果。
  - 需决策：`JWT_SECRET` / `INTERNAL_TOKEN_SECRET` 的生产取值与轮换周期；`GATEWAY_SERVICE_ACCOUNT_PASSWORD` 是否改用 Electron `safeStorage` 加密落盘；factory-service 之外是否也打开 `internal-token.enabled`（需先清理 `RecommendationClient` / Feign 等服务间直连）。
  - 环境要求：需要 zk、gateway、签发 JWT 的用户服务、Redis（refresh 黑名单）与一个有效服务账号，以及可运行 Electron 的图形环境；须另行获准。
  - 关闭条件：有一次真实链路的运行记录，未授权 REST / WS 都被拒。

- [ ] 补齐服务端事件 payload 并让前端无需回查。
  - 问题：`DigitalTwinCommandService` 广播的 `ALERT_CREATED` 只带 `alertId` / `alertCode` / `deviceCode` / `level` / `timestamp`，没有 title / description / status / createdAt，前端无法用事件直接构造完整告警，仍需局部回查 `/api/digital-twin/alerts/active`；`PRODUCTION_OUTPUT_UPDATED` 只带 `lineCode` / `output`，`stats` 也需局部回查。
  - 建议：把告警事件 payload 补齐到可直接插入列表，把产量／统计事件 payload 补齐到可直接更新 `stats`；为所有事件加 `eventId` 与 `timestamp`，前端按时间戳丢弃乱序或过期事件。
  - 需决策：事件是否采用 Sparkplug B / UNS 统一格式。
  - 环境要求：需要与上面的通用现场接入层联调；须另行获准。
  - 关闭条件：`ALERT_CREATED` 与 `PRODUCTION_OUTPUT_UPDATED` 无需回查即可更新前端状态，事件带幂等标识与新鲜度。
