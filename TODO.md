# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

本文件用中文记录任务；文件名、路径、代码标识符、命令、提交号等技术内容保留原文。

来源：2026-11-12 对 `apps/` 依据 DEVELOPMENT.md、CHECK_LIST.md 与 CODE_PINCEPLES 所做的只读合规审计。这些是既有的合规欠账，不是当次会话产生的工作。以下均为未完成项；按 DEVELOPMENT.md 第 4 步和 CHECK_LIST.md 第 2 条，每条都记录问题与建议修改方式。

2026-11-12 按**关闭每条任务所需的条件**重新分组，取代原先按主题的分节。原分四组：本机即可闭环、需人工决策、需环境／权限／外部服务、以及两者都需要。其中「本机可闭环」组的唯一任务已于 2026-11-12 完成并归档，剩余任务按后三组列出。每条都有「关闭条件」（通过验收的要求），以及「需决策」／「环境要求」行，写明具体缺什么。在 CHECK_LIST.md 验证通过前保持 `[ ]`，通过后移入下方归档区。原有的「问题」「建议」全文保留，避免丢失审计证据。

其中 PAT、性能流水线、可复现安装三条来自 2026-11-09 的 CI 清理遗留项，合规审计重写时被漏掉；2026-11-12 已对照仓库与 GitHub 重新核实。第四条遗留项（env 默认值测试的隔离性）已于 2026-11-12 修复并归档。

2026-11-15 新增一批「数字孪生与工厂自动化架构对齐」任务：把 `apps/digital-twin/` 与 ISA-95 / Purdue 分层、OPC UA + MQTT + UNS 的工业数据链路、以及边缘／云部署模式做只读对照后，把「不标准或未实现」的部分拆成 9 条，按关闭条件并入下面第 1 节（4 条）与第 3 节（5 条）。这批任务只登记，未改动任何代码。沿用 2026-11-13 的决定：不动 `apps/meta-note/`、digital-twin 继续留在 monorepo、不运行任何测试、lockfile 作为生成文件不入库；凡涉及运行验证的关闭条件都须另行获准后执行。其中「统一端口／env 口径」与既有「让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐」任务有交叉，处理时要一并决定，避免两处改动冲突。

2026-11-15 用户给出 7 条决定，覆盖这批任务：1 交付形态是纯 Electron，Web 版只是历史遗留（`5b6f2847 ECharts integrate into digital twin.` 引入了 `docker-compose.digital-twin.yml` 的 `digital-twin-frontend` 与 `k8s/digital-twin/` 的 ingress / deployment，后来包了 Electron 壳但部署资产未清理）；2 使用事件订阅，去掉定时全量轮询；3 不做分级部署、不引入云端方案，本项目就是完全面向本地工厂；4 模拟／演示数据只能出现在测试脚本或初始化演示，不能因此放弃真实数据读取与接入；5 现场协议要做通用版本（当前没有真实工厂可对接），方便后续最小修改；6 时序库若使用就必须真正接入，不允许留空依赖；7 对接 `server/` 的接口通讯必须经 zk + `server/gateway/` 鉴权，数字孪生自己内部消化的接口不需要验证。据此已关闭「事件订阅」与「分级部署」两条，把时序库改为必须实际接入、鉴权改为按 zk + gateway 与内部接口划界，并新增「补齐服务端事件 payload」一条；其余条目按新口径改写。

2026-11-15 追加两条决定：① **不允许绕开网关直连 service**——渲染层对 `server/` 的所有 REST / WS 请求必须走 `gateway:10081/factory-service/**` 并由网关校验 token，这是明确的实现要求，不是待办；② 澄清 PLC4X 1.0.0 的 JDK 口径（见第 3 节协议任务）。本轮据此直接改代码，并把已完成部分归档。

2026-11-15 鉴权链路核实结论：`server/common/src/main/resources/application-common.yml` 配置 `spring.cloud.zookeeper.connect-string=localhost:2181` 与 `dubbo.registry.protocol=zookeeper`；`server/gateway` 端口 `10081`，`discovery.locator.enabled=true`；`FactoryServiceApplication` 有 `@EnableDiscoveryClient`，`docker-compose.server.yml` 的 `x-java-service` anchor 注入 `SPRING_CLOUD_ZOOKEEPER_CONNECT_STRING=zookeeper:2181`，因此 factory-service 会注册到 zk。`UserAuthFilter` 校验 JWT 后注入 `X-User-Id` / `X-User-Role` / `X-Internal-Token`；`RequirePermissionAspect` 从 `X-User-Id` 取用户并校验权限，缺头即 401。但数字孪生前端的所有基址（`config/digital-twin.ts` 的 `localhost:10101`、`services/api/alertRule.ts` 的 `localhost:8082`、WS 的 `localhost:10101`）都直连 factory-service，既不经 gateway 也不带 token；`gateway/application.yml` 还把 `/ws/digital-twin` 放进 `excluded-path-patterns`、把 `/ws/**` 放进 `rbac-excluded-path-patterns`；`DigitalTwinAuthHandshakeInterceptor` 只检查 `X-User-Id` / `Authorization` 头是否存在，不校验 token，角色直接采信请求头。结论是**没有对上**：zk + gateway 的跨服务鉴权链路存在但被前端绕过，后端却在期待 gateway 注入的头，直连时要么 401、要么可被伪造。Electron 主进程内嵌的 `127.0.0.1:5051` / `5050` 属于数字孪生内部消化，按决定不需要验证。

2026-11-15 修正与落地（上面的「没有对上」是改造前的结论，保留作为审计证据）：按新决定已把 `config/digital-twin.ts`、`config/mes.ts`、`services/api/alertRule.ts` 与 WS 全部收敛到 `renderer/config/gateway.ts` 的 `FACTORY_SERVICE_BASE_URL`，删除 `10101` / `8082` / `8080` 三组直连端口注入；新增具名桥接 `getGatewayToken`，JWT 由主进程从 `.env` 的 `GATEWAY_TOKEN` 读取，不进渲染层 bundle。网关侧新增 `lb:ws://factory-service` 显式 WS 路由（`StripPrefix=1`），移除 `/ws/digital-twin` 与 `/ws/**` 两处免鉴权排除，`UserAuthFilter` 支持 WS 握手的 `?token=` 并校验后在转发前剥离参数；服务侧 `DigitalTwinAuthHandshakeInterceptor` 改为校验 `X-Internal-Token`（HMAC，5 秒有效）后才采信网关注入的用户头。`docker-compose.digital-twin.yml` 不再向宿主发布 factory-service 端口，客户端只暴露 `10081`。仍未闭环的是 token 签发方式、REST 侧的 `InternalTokenFilter` 全局注册与整条链路的运行验证，另立任务跟踪。

### 1. 需人工决策（必须先做出选择）

编码工作在本机即可完成，但需要先定下产品／接口／验收口径，否则修完可能被推翻。

2026-11-13 用户给出 6 条决定：监控窗口保留、按推荐的鉴权方案处理、不动原本的 meta-note、digital-twin 继续集成在 monorepo、名称统一为 `meta-note元笔记`、不做任何测试且 lockfile 作为生成文件不入库。据此本节原有的「合并监控窗口与两套 LLM 控制器」「meta-note 仓库腾空与资产处置」「本地 HTTP 服务鉴权」三条已关闭并移入归档区；`APP_PROTOCOL` 保持 `meta-note` 不变，避免影响已注册的 URL scheme 与深度链接。下面只保留仍需决策的任务。

- [ ] 把各 app 的测试覆盖补齐到项目的测试框架要求。
  - 问题：`backstage-admin` 只有 1 个测试文件（`tests/env.spec.ts`），却对应 93 个手写 `src/apis/` 模块；`apps/client` 有 9 个测试文件，全在 `lib` 层，没有组件 / 页面覆盖，且 e2e 套件被禁用；`digital-twin/system-management` 有 7 个单测文件，另有 `playwright.config.ts` 和一个 `src/renderer/e2e/scene.spec.ts`，没有响应式 / 视觉 / 性能覆盖。
  - 建议：为每个 app 的 API 模块与关键页面 / 屏幕补测试，让 e2e 用例能在 CI 运行，并断言行为而不是「没有抛异常」。
  - 需决策：先确定验收口径。DEVELOPMENT.md 只写了「add tests matching the project's test framework」；若要求数值化的覆盖率门槛，必须先定目标百分比。行为断言本身不依赖它，可以先写。
  - 关闭条件：三个 app 都补上 API 模块与关键页面 / 屏幕的测试，断言行为而非 `not.toThrow()`，让 e2e 用例可在 CI 运行，并重跑各 app 的 typecheck 与测试。

- [ ] 清理三个客户端里的代码原则违规项。
  - 问题：DEVELOPMENT.md `:82-85` 要求代码文本一律英文（i18n 除外）、函数不超过 20 行、不吞异常、无隐藏共享状态、使用卫语句。对 `src/` 的静态扫描发现中文文本：backstage-admin 5751 行、client 891 行、digital-twin 509 行（修复时须把 i18n 文案与注释/日志区分开）；catch 块分别为 581 / 15 / 65 个，其中若干只记录日志就返回；审计还报告 digital-twin 控制器里约 227 行的函数（修复前重新测量）。
  - 建议：按 app 逐个处理：翻译注释、日志与错误文案（只保留 i18n 资源作为中文），把「记录后忽略」的 catch 改成显式错误状态或重新抛出，拆分超过 20 行的函数。每轮之后重跑该 app 的 typecheck 与测试。
  - 需决策：界定哪些中文字符串属于 i18n 资源（只有这些可以保留）。5751 / 891 / 509 行里混有 i18n 文案、注释、日志和错误字符串，批量改写前须先商定基于路径的分类规则。
  - 关闭条件：按 app 逐个把注释 / 日志 / 错误文案改为英文，把记录后忽略的 catch 改成显式错误状态或重新抛出，拆分超过 20 行的函数（先重新测量那个约 227 行的控制器），每轮都通过 typecheck 与测试。

- [ ] 补齐三个客户端的响应式与可访问性缺口。
  - 问题：对 `src/` 的静态扫描：`@media` 只出现在 backstage-admin 的 4 个文件，另外两个 app 为 0；`aria-*` 命中数 1 / 0 / 21，`useWindowDimensions` / `Dimensions` 命中数 0 / 1 / 0；`apps/backstage-admin/src/styles/index.scss:32,37` 用 `outline: none` 移除了焦点轮廓；`apps/client` 没有无障碍属性，也没有 ErrorBoundary，并把方向锁死为竖屏（`app.config.ts:9`、`android/app/src/main/AndroidManifest.xml:19`）；`digital-twin` 没有媒体查询。
  - 建议：为每个 app 增加断点与弹性布局，恢复可见焦点态，补充语义标签 / `aria-*` 或 `accessibilityLabel` / `accessibilityRole`，并增加 ErrorBoundary 与加载失败时用户可感知的降级表现（FRONTEND_PRICEPLES 第 7、8 节）。
  - 需决策：提供设计基准 / 断点取值（仓库里没有设计稿）。若没有，默认采用通用断点集，并以可评审的变更提交。
  - 关闭条件：三个 app 都加上断点与弹性布局、恢复可见焦点态、补充语义标签 / `aria-*` / `accessibilityLabel` / `accessibilityRole`、加入带用户可见降级的 ErrorBoundary，并用各 app 的检查验证。

- [ ] 让 `apps/backstage-admin` 的 API 客户端与 DEVELOPMENT.md 第 3 节保持一致。
  - 问题：`package.json:18` 定义了 `generate:api`，`tools/generate-api.cjs` 调用共享的 `apps/tools/OpenApiToTS.js`，其默认输出为 `src/generated/api`，但该目录不存在；应用实际调用的是 `src/apis/` 下 93 个手写模块。
  - 建议：要么运行生成器并把调用方迁移到生成的客户端，要么在 DEVELOPMENT.md 与 app README 中记录这一有意偏离（包括为何保留手写模块、以及如何保持同步）。
  - 需决策：在建议的两条路线中二选一：运行生成器并迁移 93 个手写模块，或记录有意偏离。需与下面「对齐 DEVELOPMENT.md 第 3 节」一起决定，否则两处改动会冲突。
  - 环境要求（仅生成器路线）：后端 OpenAPI 文档可访问，且生成器工具链可用。
  - 关闭条件：要么迁移调用方并产出 `src/generated/api/`，要么把偏离说明落到 DEVELOPMENT.md 与 app README，并写明同步机制。

- [ ] 让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐。
  - 问题：`DEVELOPMENT.md:49-58` 要求 `.env.development` 定义 `NEXT_PUBLIC_BACK_API_HOST` / `NEXT_PUBLIC_BACK_API_DOC_HOST`，且两个 app 都输出到 `src/generated/api/`。实际上 `backstage-admin/.env.development` 只设置了 `VITE_BLOCKCHAIN_ENABLED=true`，其生成器读取 `.env,.env.local` 中的 `VITE_BASE_SERVER_URL` / `VITE_API_DOC_HOST`；`apps/client` 把 `NEXT_PUBLIC_*` 变量放在 `.env`，并有 1536 个生成文件；`digital-twin` 正确地没有 `generate:api`，只有 5 个手写类型文件。文档与两处实现都不一致。
  - 建议：选定一套 env 文件 / 变量约定，同步更新三个 app 与 `.env.example` 文件，并重写第 3 节描述各 app 的真实行为，包含 digital-twin 的例外。
  - 需决策：在三个 app 间选定唯一一套 env 文件 / 变量约定（`NEXT_PUBLIC_*` 还是 `VITE_*`，`.env.development` 还是 `.env`）。需与上面「让 backstage-admin API 客户端与第 3 节一致」一起决定。
  - 关闭条件：更新三个 app 与 `.env.example` 文件，重写 DEVELOPMENT.md 第 3 节，描述各 app 的真实行为并包含 digital-twin 的例外。

- [ ] 清理 Web 版历史遗留部署资产，确认纯 Electron 的本地工厂单档形态。
  - 问题：交付形态已定为纯 Electron、面向本地工厂，但 `5b6f2847 ECharts integrate into digital twin.` 引入的 Web 部署资产仍在：`k8s/digital-twin/{deployment,frontend,ingress,service}.yaml`（`ingress.yaml` 的 host 为 `digital-twin.meta-web-three.local`，`frontend.yaml` 是 2 副本 nginx Deployment），全仓无任何引用；`apps/digital-twin/system-management/docker/Dockerfile` 仍以 nginx:alpine 托管 vite 产物（EXPOSE 3000），`docker/nginx.conf.d/default.conf` 把 `/api/` 代理到 `host.docker.internal:8080`、`/ws` 到 `:8082/ws`，与 factory-service 的 `10101` 不符。2026-11-15 已从 `docker-compose.digital-twin.yml` 删除 `digital-twin-frontend` 服务段（它注入的 `VITE_*` 是构建期变量、builder 阶段收不到，本来就不生效），Web 版不再是部署入口。
  - 建议：确认后删除 `k8s/digital-twin/` 四个 yaml；把 `docker/Dockerfile` 与 `nginx.conf` 明确标注为仅供 CI 构建校验、不作为交付形态（删除前先确认 CI 是否引用）；README 写明唯一入口是 Electron 桌面端，渲染层依赖 `window.desktop` 的本地能力，不存在 B/S 降级形态。
  - 需决策：是否删除 `k8s/digital-twin/` 与 nginx 资产。这四项是 git 跟踪文件，删除后仍可从历史恢复，但需确认没有其他部署脚本依赖。
  - 关闭条件：仓库内只剩 Electron 一种交付形态，遗留 Web 部署资产被删除或有明确用途标注，README、compose 与 CI 引用口径一致。按本轮决定不运行测试。

- [ ] 统一本地工厂单档部署的端口、环境变量与 compose 口径。
  - 问题（2026-11-15 已修掉的部分）：`docker-compose.digital-twin.yml` 原先把后端映射 `8082:10101`，现已改为 `ports: []`（只留 gateway `10081`）；渲染层三组直连端口 `10101` / `8082` / `8080` 已从 `vite.config.ts` 与 `.env.example` 删除，改为 `VITE_GATEWAY_URL` / `_HOST` / `_PORT` 与 `GATEWAY_TOKEN`；`.env.example` 里 `VITE_VOICE_API_URL` 默认 `5051` 与实际 `8000` 不符的问题也已修正；`SPRING_KAFKA_BOOTSTRAP_SERVERS` 改为 `${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}`，本机差异可放 `.env`。**仍存在**：`apps/digital-twin/system-management/docker/docker-compose.yml` 是第二套口径（`VITE_API_URL=http://localhost:8080`、`VITE_WS_URL=ws://localhost:8082/ws`），`docker/nginx.conf.d/default.conf` 把 `/api/` 代理到 `host.docker.internal:8080`、`/ws` 到 `:8082/ws`，两者都是已废弃 Web 版的残留；`VITE_MQTT_URL=ws://localhost:9001` 指向的 broker 从未启动；`depends_on: [kafka, redis, zookeeper]` 的三个服务都不在本文件里，需叠加根 `docker-compose.yml`（它 include 了 `docker-compose.env.yml` / `docker-compose.server.yml`），而 `docker-compose.env.yml` 的 kafka 用 `network_mode: host`，容器里解析不到 `kafka:9092`；`8080` 仍被 `WHISPER_URL` 与内置 `llama-server` 同时占用。
  - 建议：删除或明确标注废弃的 nginx / 第二套 compose 资产；把唯一的本地工厂端口表写进 README（gateway `10081`、factory-service `10101` 仅容器内、Whisper、llama-server、`5051` / `5050` / `8000`）；为 kafka 的 host 网络模式确定一个容器内可达的地址（或改 compose 网络口径）；补 MQTT broker 服务或删掉该变量；把 `WHISPER_URL` 与 `llama-server` 的 `8080` 错开。
  - 需决策：定唯一端口分配表与 compose 入口（确认以根 `docker-compose.yml` + `docker-compose.digital-twin.yml` 叠加为准）；env 命名规则需与「让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐」一并决定；确认是否直接删除 nginx 与第二套 compose。
  - 关闭条件：只剩一份本地工厂口径，所有 `VITE_*` 在构建期生效且登记在 `.env.example`，容器起来后 3D 页面 / 告警规则 / MES 面板经 gateway 连到正确地址，`8080` 不再冲突，kafka 可解析。本轮只做静态核对，不运行测试。

（「用事件订阅替代轮询」与「提供分级部署档位」两条已按 2026-11-15 的决定关闭，详见归档区；分级部署／云端方案明确不做，只保留本地工厂单档。）

### 2. 需环境 / 权限（设备、账号或外部服务）

剩余的代码改动很小、也能在本机完成，但关闭该项需要普通检出环境里没有的运行时。

- [ ] 重新启用 `apps/client` 的 e2e 套件并修复其配置。
  - 问题：`e2e/setup.ts:28` 曾赋值 `global.describe = global.describe.skip`，导致所有 detox 用例永久失效；`e2e/jest.config.js` 是裸对象字面量、缺少 `module.exports`，`require()` 报 `Unexpected token ':'`。这两处代码缺陷已于 2026-11-12 修复：跳过现在需要显式 `E2E_SKIP=1`，配置已导出模块且 `setupFilesAfterEnv` 指向 `<rootDir>/setup.ts`。仍未完成：没有任何 detox 用例在模拟器或真机上跑过，也没有 CI job 运行该套件。
  - 补充发现与修复（2026-11-12 发现，2026-11-15 已在本机修复）：还有 4 处既有缺陷会阻断任何真机运行，上面没有记录。`e2e/setup.ts` 导入的 `detox-adapter-jest` 与 `detox-matchers` 都未在 `package.json` 声明（全仓仅出现在该文件的引用里）；`e2e/jest.config.js:5` 的 reporters 含未声明的 `detox-runner-reporter`；`e2e/detox.config.js` 用的是 Detox 20 之前的 `composition` / `session` 结构，且不在默认 `.detoxrc.js` 路径，第 18 行还有非 ASCII 字段 `油脂: 'keep'`；`e2e/app.spec.ts:3-13` 导入 `../pages`，而页面对象实际在 `e2e/pages/index.ts`。现已把配置重写为 Detox 20 的 `apps` / `devices` / `configurations` 结构，默认设备改为 Android（本机 `yarn android` 的 `app-debug.apk` 构建产物已存在），删掉非 ASCII 字段与未声明包；jest 改用 `detox/runners/jest/*` 的 `globalSetup` / `testEnvironment` / `reporter`，`setup.ts` 只保留超时与显式 `E2E_SKIP` 开关；`package.json` 三个 e2e 脚本显式带 `--config-path e2e/detox.config.js -c android.emu.debug`（真机用 `-c android.attached.debug`）。另修掉 4 类会阻断运行的旧 spec API：`device.back()` → `device.pressBack()`（8 处）、`reload()` → `device.reloadReactNative()`、`cleanup()` 的 `afterAll` 交给 testEnvironment、`launchApp({ fromIcon: true })` → `launchApp({ newInstance: false })`；`node_modules/detox/detox.d.ts` 的 `DetoxExportWrapper` 并不导出 `init` / `cleanup` / `reload`。静态验证：两个 JS 配置 `node --check` 通过且可 `require()`，`detox/runners/jest/*` 与 `jest-junit` 均可解析，`package.json` 为合法 JSON，三个 TS 文件用 `@babel/parser` 解析通过，`git diff --check` 干净；按 2026-11-13 的决定未运行任何测试。
  - 建议：至少运行一条真实用例（`E2E_SKIP` 不设置，CI 或有记录的手动真机检查均可）并记录结果，之后才能移除本项。不要重新引入全局跳过。
  - 环境要求：需要 Android SDK 与模拟器镜像（本机 Windows 可行，AVD 名通过 `DETOX_AVD_NAME` 指定，默认 `Pixel_7_API_34`），或一台已连接的 Android 真机（`-c android.attached.debug`，`DETOX_ADB_NAME` 指定设备名）。配置改为 Android 后不再需要 macOS + Xcode。要长期保持套件可用，还需要具备同等运行时的 CI job，或一份有记录的手动真机检查。
  - 关闭条件：代码缺陷已全部修复；剩余只需在 `E2E_SKIP` 不设置的情况下，于模拟器或真机上至少跑通一条真实用例并记录结果（必要时修正页面对象与真实界面不匹配的选择器）。

### 3. 需决策 + 环境（两者缺一不可）

每一项都既需要明确的批准或接口决策，又需要外部资源（设备构建、账号权限、可达服务或 CI 运行时）。

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
  - 问题：第 6 节（`:49-50`）要求可执行的视觉质量检测机制和量化差异标准；第 9 节（`:68-69`）要求自动化覆盖结构正确性、响应式布局、视觉一致性、性能指标与基础可访问性。`apps/` 下没有任何视觉 / 截图 / a11y / 性能测试文件。现有性能断言只检查 `not.toThrow()`（如 `AlertFlow.test.tsx:695`、`RealTimeDataDisplay.test.tsx:226`）。
  - 建议：增加可运行的截图 diff / 视觉回归任务并给出明确的像素或感知阈值，把 a11y 与性能预算接入 CI；按 app 选工具（两个 web 前端用 Playwright，Expo app 用设备矩阵截图运行器）。
  - 需决策：确定每个 app 的工具、视觉差异的数值阈值（像素或感知）、基线截图由谁批准，以及 a11y / 性能预算的具体数值。FRONTEND_PRICEPLES 第 6、9 节要求量化标准，但没有给出数值。
  - 环境要求：两个 web 前端需要 Playwright 浏览器二进制与基线图片，Expo app 需要设备矩阵截图运行器，还需要能存储并比对基线的 CI job。
  - 关闭条件：落地带明确阈值的可运行截图 diff 任务，把 a11y 与性能预算接入 CI，并给出一次通过、一次故意失败的运行作为证据。

- [ ] 撤销已泄露的 classic PAT 并从 git 历史中清除。
  - 问题：提交 `3c07b882`（2026-05-26）在 `package.json` 的 `repository.url` 里存了明文 `ghp_…` token；虽然当前 `package.json`、`.git/config` 与 `remote.origin.url` 都已清理，且 `lefthook.yml:23-28` 现在会拒绝新增行里的 token，但该值仍可从对象库读出。这个历史 token 已返回 HTTP 401，但从未被撤销或清除。本机不存在 `~/.git-credentials`（`credential.helper=manager` 把凭据存在 Windows 凭据管理器里），所以 2026-11-09 报告的那个仍在使用的 classic PAT 必须到 GitHub Settings 里核查并撤销，而不是删某个文件。2026-11-12 新增了 CI 密钥扫描（`.github/workflows/secret-scan.yml`）：每次 push 与 PR 阻断新引入的密钥，手动或定时运行时输出全历史报告。
  - 建议：仍未完成：在 GitHub Settings 删除 / 撤销 classic PAT，换成细粒度 PAT 或操作系统凭据管理器，并用 `git filter-repo` / BFG 清除历史值，配合协同 force-push 与清除后的轮换。若不接受重写历史，需明确记录所接受的风险。只有在账号侧动作确认后才能移除本项。
  - 需决策：批准（或拒绝）重写 git 历史。`filter-repo` / BFG 会改变每一个提交哈希，影响所有协作者、fork、未合并 PR 与 CI 引用，需要协同 force-push。若拒绝，替代方案是明确记录所接受的风险。
  - 环境要求：需要 GitHub 账号 Settings -> Developer settings 权限来撤销 classic PAT、仓库管理员权限执行 force-push、与所有协作者协调、以及清除后的凭据轮换。正在使用的凭据在 Windows 凭据管理器里，不在 `~/.git-credentials`。
  - 关闭条件：在 GitHub Settings 确认撤销，换成细粒度 PAT 或操作系统凭据管理器，通过协同 force-push 清除历史值（或记录所接受的风险），并重跑一次全历史密钥报告。

- [ ] 把性能流水线指向真实的 staging 目标。
  - 问题：`GET /repos/lotaway/meta-web-three/actions/variables` 返回 `total_count: 0`，因此 `vars.PERF_BASE_URL` 未设置，每次运行测的都是内置的 `127.0.0.1:18087` mock；#163–#165 三次运行虽然全绿，但从未触达真实服务。2026-11-12 把 schedule 分支里无条件的 `use_mock = True`（原 `:127`）改为统一的 `use_mock = not base`，mock 运行现在会打 warning，报告也标注 MOCK DATA，避免把 mock 数字当成 staging 结果。这些代码缺陷已修复；仓库变量仍未设置，也没有任何一次运行针对真实服务验证过。
  - 建议：把 `PERF_BASE_URL` 仓库 / 环境变量设为可达的 staging 网关，重跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL 且 `use_mock=false`，之后才能移除本项。
  - 需决策：批准对 staging 做压测，提供目标 URL 与测试窗口，并决定每日定时任务是否也打它、还是只允许手动运行。
  - 环境要求：需要从 `ubuntu-24.04` runner 可达的 staging 网关、设置 `PERF_BASE_URL` 所需的仓库 / 环境变量写权限，以及 `workflow_dispatch` 权限。
  - 关闭条件：设置变量后跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL、`use_mock=false`，且没有 MOCK DATA 标注。

- [ ] 为 `apps/digital-twin/system-management` 补齐 Electron 运行时冒烟检查与 CI 原生 ABI 覆盖。
  - 问题：`d10f4015` 新增了 `.github/workflows/system-management-check.yml`（run #1 成功），修正了构建上下文 / Dockerfile / compose 路径，并在 CI 里加入 `npm run build:native`、`npm run typecheck` 和 `npm run test:run`，解决了此前「没有 CI 构建它」的缺口。仍未完成：`docker/Dockerfile:20-24` 与 workflow `:49` 都用 `npm install --legacy-peer-deps`，两者都不执行 `npm run rebuild:sqlite3`（而 app 声明了 `better-sqlite3 ^11.0.0`），也都未启动 Electron（`docker/Dockerfile:4` 与 workflow `:48` 的 `ELECTRON_SKIP_BINARY_DOWNLOAD=1`）。因此原生 ABI 回归从不在运行时被检验。渲染进程隔离已于 2026-11-13 加固，并用两个隐藏 `BrowserWindow` 的受控冒烟验证了隔离配置与 preload 注入（见归档区），但那不是加载真实业务 UI 的完整回归。
  - 2026-11-13 决定：lockfile 属于生成文件，明确不放入版本控制（根 `.gitignore` 与本 app `.gitignore` 已忽略，仓库中无被跟踪的 lockfile）。因此不再有「提交 lockfile / 切回 `npm ci`」这条路线，也不再把 lockfile 缺失当作缺陷；可复现安装改用 `.npmrc` 固定 registry 加显式依赖版本约束。用户同时明确「不要做任何测试，环境是没问题的」，冒烟与端到端验证本轮一律豁免，本项继续保持 `[ ]`。
  - 建议：用 `.npmrc` 固定 registry，增加 `npm run rebuild:sqlite3` 的 CI 步骤以及无头 Electron 启动 / 冒烟检查（或记录所需的手动真机检查）；需要运行验证时再单独取得授权。
  - 环境要求：CI 里需要无头 Electron 运行时（xvfb，并取消 `ELECTRON_SKIP_BINARY_DOWNLOAD=1`）、通过 `.npmrc` 锁定 registry，以及两个 BrowserWindow 的真机 / 图形界面运行来取得冒烟结果。2026-11-13 的受控冒烟已覆盖「两个窗口的隔离配置」这一部分，仍缺 CI 无头冒烟与加载真实业务 UI 的完整回归。
  - 关闭条件：加入 `npm run rebuild:sqlite3`、加入无头 Electron 启动 / 冒烟步骤，并在获准运行验证后记录两个窗口的一次真实运行。

- [ ] 按「演示数据仅用于初始化／测试」的边界重建设备数据通路。
  - 问题：`DigitalTwinSimulator.java` 标 `@Profile("dev")` 且用 `@Scheduled(initialDelay = 5000, fixedDelay = 10000)` 随机改设备状态；`DigitalTwinDevDataInitializer.java` 标 `@Profile("dev")` 注册 8 台演示设备（AGV / ROBOT / PLC / CONVEYOR）；`MqttConfig.java` 默认 `mqtt.enabled:false`。而 `docker-compose.digital-twin.yml` 设 `SPRING_PROFILES_ACTIVE=prod`，覆盖了 `application.yml:7` 的默认 `dev` —— 容器启动后既没有演示数据也没有采集数据，`/api/digital-twin/devices` 为空。按 2026-11-15 第 4 条决定，模拟／演示数据只能出现在测试脚本或初始化时提供给用户做演示，绝不能因此放弃真实数据的读取与接入。
  - 建议：把模拟器与 8 台演示设备从 `dev` 抽到显式的 `demo` 初始化器（或移进 `src/test` 的 fixture），仅在用户选择「演示模式」时写入一次，不再定时改写运行时状态；`prod` 必须接到下面「协议无关的通用现场接入层」的真实数据源；MQTT 开关由 compose 显式注入并在 prod 打开。状态实体补 `source`（REAL / DEMO）/ `quality` / `timestamp`，前端在演示模式下明确标注数据非真实。
  - 需决策：演示模式的触发方式（env 开关 / 首次启动引导 / 设置页），以及演示数据与真实数据是否允许共存。
  - 环境要求：需要可达的 MQTT broker 与至少一个数据源（真实设备或 OPC UA / PLC 仿真器）才能验证 prod 通路；按本轮决定不运行验证，须另行获准。
  - 关闭条件：prod 能从真实数据源读到设备与遥测，`demo` 只做一次性初始化、不参与运行时状态变更，测试脚本里的模拟数据可复现，前端能区分 REAL / DEMO 与数据新鲜度。

- [ ] 建设协议无关的通用现场接入层（当前没有真实工厂，目标是后续最小修改）。
  - 问题：全仓没有 OPC UA / Modbus / 西门子 S7 等现场协议实现；`Device.mqttTopic` 只是实体字段；`MqttTelemetrySubscriber.java` 只订阅 `MqttConfig` 默认的裸前缀 `mes/equipment/`，没有 Sparkplug B、没有按 ISA-95 `Enterprise → Site → Area → Line → Cell → Device → Tag` 组织的 UNS、没有单位 / 时间戳 / quality 归一化，也没有断网缓存与补发。目前不存在可用的真实数据读取路径。
  - 2026-11-15 标准库调研结论：Apache PLC4X（2026-09-07 发布 1.0.0，Apache-2.0）用一套 `PlcDriverManager` API 覆盖 S7 / Modbus TCP / OPC UA / EtherNet-IP / ADS / BACnet / KNX，改连接串与点位地址即可换协议，最符合「通用版本 + 最小修改」。Eclipse Milo（`org.eclipse.milo:sdk-client`）是只做 OPC UA 时的成熟备选；MQTT 已有 `org.eclipse.paho.client.mqttv3:1.2.5`；Sparkplug B 可用 Eclipse Tahu 做 payload / topic 规范。PLC4X 不负责 MQTT 发布，读到值后要自行编码交给 Paho。
  - 2026-11-15 JDK 口径澄清（用户提问「api 模块是 Java 17，为什么驱动要求 Java 21」）：官方迁移指南原文是「The `plc4j-api` module is intentionally held at Java 17, so an alternate driver implementation can still target 17. Everything else — the SPI, the drivers, the tools — needs 21」；1.0.0 release notes 同样写明「Dropped support for Java 11, new baseline Java version is Java 21」并单独注明 API 模块保留 17。也就是说 Java 17 **只是接口 jar 的字节码版本**（class major 61），目的只是让第三方能自行实现驱动而不必上 21；官方 SPI / 驱动 / 工具均以 `--release 21` 编译（major 65），JDK 17 无法加载。本仓库 `server/pom.xml:50` 的 `java.version=17` 只是我们自己的编译目标，管不到依赖 jar 的字节码版本；即便用 JDK 21 以 `--release 17` 编译本项目，只要运行时是 17 就仍加载不了官方驱动。结论：要用 PLC4X 1.0.0 的官方驱动，运行时必须升到 JDK 21（Spring Boot 3.5.4 兼容，源码仍可按 17 编译）；否则只能锁 0.13.1（Java 11 基线）或改用 Milo + Paho。依据：[PLC4J 0.13.1→1.0.0 迁移指南](https://plc4x.apache.org/plc4x/latest/users/migrations/0.13.1-to-1.0.0/plc4j.html)、[1.0.0 release notes](https://github.com/apache/plc4x/releases/latest)、[CAMEL-25617](https://www.mail-archive.com/issues@camel.apache.org/msg136778.html)（"PLC4X 1.0 compiles its drivers with release 21; only plc4j-api stays on 17"）。
  - 建议：先做协议无关的 SPI：定义 `TelemetrySource`（connect / read / subscribe / close）与配置化点位映射（连接串、tag 地址、单位、quality、采样周期），把现有 MQTT 订阅做成默认实现，再以可选 profile 挂载 PLC4X 驱动（S7 / Modbus / OPC UA）或 Milo；对外统一发布到按 ISA-95 层级组织的 UNS topic，边缘侧做死区过滤与断网缓存补发。换协议只改配置与一个适配器，业务代码不动。
  - 需决策：是否把 JDK 升到 21 以用 PLC4X 1.0.0，还是锁 0.13.x / 只做 MQTT + Milo；首批支持的协议集合；UNS 层级与 Sparkplug B 是否采用。
  - 环境要求：需要 OPC UA 服务器或 PLC 仿真器（如 PLC4X mock driver）、MQTT broker、OT 网络 VLAN 与证书。
  - 关闭条件：`TelemetrySource` 接口与配置化点位映射落地，至少两种来源（MQTT + 一种仿真 PLC/OPC UA）能把带 quality 与时间戳的点位送进 `factory-service` 并在 3D 页面显示，新增协议只需加一个适配器与配置。运行验证须另行获准。

- [ ] 实际接入 InfluxDB 时序库，不允许留空依赖。
  - 问题：`pom.xml:46-47` 声明 `com.influxdb:influxdb-client-java:7.2.0`，但 `src/main/java` 里零使用；`TelemetryRecord` / `ScadaTelemetryRecordDO` 走关系型 mapper，没有时序读写与历史回放；`ScadaController` 的 `/api/mes/scada/telemetry/{code}/range` 因此拿不到真正的时序数据，追溯与趋势面板缺数据底座。按 2026-11-15 第 6 条决定，既然声明了就要接入使用，不能放空，也不接受直接删掉依赖了事。
  - 建议：新增 `InfluxTelemetryStore`（`InfluxDBClientFactory.create(url, token, org, bucket)` + `WriteApiBlocking` / `QueryApi`），在 `MqttTelemetryService` 的 `scadaDomainService.ingestTelemetry(...)` 路径与 `POST /api/mes/scada/telemetry/ingest` 写入 point（measurement=`telemetry`，tag=`equipmentCode`，field 为解析后的指标，timestamp 用采集时间）；把 `getTelemetryByRange` 改走 InfluxDB 范围查询，最新值查询仍可走关系库；连接参数（url / token / org / bucket）通过 `application-common*.yml` 注入并在 compose 起一个 InfluxDB 服务；写入失败要显式报错而不是静默吞掉。
  - 需决策：measurement / tag 命名与保留周期；InfluxDB 版本与是否启用鉴权 token；是否保留关系库中的遥测表做双写。
  - 环境要求：需要 InfluxDB 实例（目前 `docker-compose.digital-twin.yml` 只有 kafka / redis）与历史数据量评估。
  - 关闭条件：遥测可写入 InfluxDB 并按设备 / 时间范围查询返回，`/api/mes/scada/telemetry/{code}/range` 有真实数据来源，依赖不再是空声明。运行验证须另行获准。

- [ ] 明确「数字影子 / 闭环孪生」边界，并实现受控写回（如需）。
  - 问题：`ScadaController.java:58` 有 `POST /api/mes/scada/commands`，`MqttCommandPublisher.java` 能把指令发到 broker，但写回完全由人工触发，没有自动或受控闭环；按 Kritzinger 的分类，只有单向展示的是 digital shadow，而当前连展示数据都来自 `DigitalTwinSimulator`。README 却一直按「数字孪生」描述。按第 4 条决定，演示数据不能替代真实接入，这条边界要在真实数据源接通后再定。
  - 建议：若需要闭环，做「建议 → 人工确认 → 白名单点位写回 → 审计」链路，控制逻辑与安全联锁仍留在 PLC；若不需要，把产品命名与 README 统一为「数字影子 / 可视化监控」，不提供写回。
  - 需决策：是否允许写回产线、审批流与白名单点位范围、产品命名是否调整。
  - 环境要求：需要可写测试点位、PLC 侧安全联锁与写操作审计存储。
  - 关闭条件：闭环链路可用且有审计记录，或文档明确单向边界并统一命名。

- [ ] 确定 `GATEWAY_TOKEN` 的签发方式并验证 zk + gateway 全链路。
  - 问题：前端收敛到网关、网关 WS 鉴权与服务侧握手校验已按新决定实现（见归档区与本文件开头的修正段落），但仍有三处未闭环。① token 来源：目前只是把平台签发的 JWT 手工写进 `.env` 的 `GATEWAY_TOKEN`，由主进程读取，没有登录、刷新与过期处理；本应用没有登录界面。② `server/common` 里已有 `InternalTokenFilter`（校验 `X-Internal-Token`）却没注册进 `FactoryServiceApplication`（`scanBasePackages` 不含 `com.metawebthree.common.filter` / `.utils`），所以 REST 侧目前只靠 `RequirePermissionAspect` 依赖 `X-User-Id`；WS 握手已强制校验内部令牌，但容器内绕过网关直连未加权限注解的 REST 接口仍可能成功。③ 整条链路按决定没有运行验证：zk 注册、`discovery.locator` 的 `/factory-service` 前缀剥离、显式 `lb:ws://factory-service` 路由、WS 自定义头是否透传、5 秒内部令牌窗口，都只有静态核对。
  - 建议：① 定 token 口径：复用 `mall-user-service` SSO 登录 / 本地工厂离线账号 / 运维预置长期 JWT 三选一；若走登录，需再加一个具名桥接写入并实现刷新。② 把 `InternalTokenUtil` 与 `InternalTokenFilter` 显式注册到 `FactoryServiceApplication` 的 `@Import`，让所有 REST 也拒绝非网关来源（保留 `/actuator/**` 等排除路径）。③ 获准后做一次端到端：未带 token 的 REST / WS 均 401，带有效 JWT 能拿到设备列表与 5 类事件，并记录结果。
  - 需决策：token 签发与刷新口径；是否对所有 REST 强制内部令牌；factory-service 的角色与 `RequirePermission` 权限映射如何分配。
  - 环境要求：需要 zk、gateway、签发 JWT 的用户服务与一个有效 token，以及可运行 Electron 的图形环境；按本轮决定不运行验证，须另行获准。
  - 关闭条件：token 来源明确并落地，未授权 REST / WS 都被拒，Electron 内部 `5051` / `5050` 边界在文档写明，并有一次真实链路的运行记录。

- [ ] 补齐服务端事件 payload 并接线未使用的实时通道。
  - 问题：前端已按第 2 条决定改为事件驱动（见归档区），但 `DigitalTwinCommandService` 广播的 `ALERT_CREATED` 只带 `alertId` / `alertCode` / `deviceCode` / `level` / `timestamp`，没有 title / description / status / createdAt，前端无法用事件直接构造完整告警，目前仍需局部回查 `/api/digital-twin/alerts/active`；`PRODUCTION_OUTPUT_UPDATED` 只带 `lineCode` / `output`，`stats` 也需局部回查。此外 `services/mqtt.ts` 的 MQTT 客户端、`useAlertNotification` 的告警 WS 与 `WEBSOCKET_URL`（本地 5050）写好但无调用方。
  - 建议：把告警事件 payload 补齐到可直接插入列表，把产量／统计事件 payload 补齐到可直接更新 `stats`；为所有事件加 `eventId` 与 `timestamp`，前端按时间戳丢弃乱序或过期事件；确认 MQTT 通道与本地 5050 是否属于「数字孪生内部消化」，接线或删除，不留死代码。
  - 需决策：事件是否采用 Sparkplug B / UNS 统一格式；MQTT 通道保留还是移除。
  - 环境要求：需要与上面的通用现场接入层联调；按本轮决定不运行验证，须另行获准。
  - 关闭条件：`ALERT_CREATED` 与 `PRODUCTION_OUTPUT_UPDATED` 无需回查即可更新前端状态，事件带幂等标识与新鲜度，未接线通道被接入或删除。

## 已完成任务（归档）

仅用中文记录。每项归档前均已按 CHECK_LIST.md 验证；2026-11-13 用户明确「不要做任何测试，环境是没问题的」，故本轮新增的三条归档只做静态类型检查、空白检查与预提交守卫，运行验证按用户决定豁免。
按 DEVELOPMENT.md 第 4 步，通过的任务从可执行列表移除，并在此以勾选项记录。
已完成、但所属大任务仍需人工决策或外部环境才能关闭的代码子项，也在此单独以 `[x]` 登记，并在条目里注明该大任务为何仍保持 `[ ]`。

- [x] 清理 TODO.md 中已完成与无法核实的任务。提交：`57d1db9a`。
- [x] 在 CI 与 Docker 中于 typecheck 前生成原生类型；修正 digital-twin 构建上下文、Dockerfile、compose 路径并新增检查 workflow。提交：`d10f4015`。
- [x] 修复预提交守卫在 Windows 下失败的问题；用暂存探针验证 hooks。提交：`ccb508d1`。
- [x] 用纯函数 `resolveEnv(vars)` 与显式输入测试，使 `apps/backstage-admin` 的 env 默认值测试具备隔离性。报告中的本机 1/5 失败未能复现（实测 5/5 通过）；真正的缺陷是依赖外部环境 `.env`。已在默认与非默认 env 文件下验证 8/8 通过，并通过 `vue-tsc --build`。提交：`efc88769`。
- [x] 停止在 `apps/backstage-admin` 中持久化明文登录密码：从 `UserInfo`、store 初始状态与 `userLogin` 中移除 `password`，登录页不再回填密码；新增 `src/utils/user-storage.ts` 清理历史落盘凭据，persist 改为非敏感字段白名单并在 hydrate 前执行清理。新增 12 个单测覆盖清理函数、store 行为与真实插件持久化链路；`npx vitest run` 20/20 通过，`npx vue-tsc --build` 通过。提交：`4490305b`。
- [x] 修复 `apps/client` e2e 套件的两处阻断性配置缺陷：`e2e/setup.ts` 不再无条件跳过全部用例（改为仅 `E2E_SKIP=1` 时跳过），`e2e/jest.config.js` 补上 `module.exports` 并让 `setupFilesAfterEnv` 指向 `<rootDir>/setup.ts`。已核对配置文件可被 `require()` 解析。这是「重新启用 `apps/client` e2e 套件」任务里已完成的代码子项；该任务仍需补齐缺失依赖、按 Detox 20 重写配置、修正 spec 导入路径并在真机跑通，故保持 `[ ]`。提交：`efc88769`。
- [x] 新增 `.github/workflows/secret-scan.yml`：push / PR 用 gitleaks 阻断新增泄露，手动或定时运行以固定版本 `v8.30.1` 输出全历史报告；CI `Secret Scan` run #1（`43200517750`）成功。这是「撤销已泄露的 classic PAT 并从 git 历史中清除」任务里已完成的代码子项；账号侧撤销与历史清除仍需人工执行与授权，故该任务保持 `[ ]`。提交：`efc88769`。
- [x] 修正 `.github/workflows/performance-test.yml`：schedule 分支不再无条件 `use_mock = True`，统一为 `use_mock = not base`，mock 运行输出 warning 并在报告标题 / Environment 标注 `MOCK DATA`；两段内联 Python 已通过 AST 与 4 个场景校验。这是「把性能流水线指向真实 staging 目标」任务里已完成的代码子项；`PERF_BASE_URL` 仍未设置、也未针对真实服务验证过，故该任务保持 `[ ]`。提交：`efc88769`。
- [x] 加固 `apps/digital-twin/system-management` 的 Electron 渲染进程隔离，同时保留 AI Agent 所需的文件读写 / 查询 / 显示能力。按 2026-11-13 的决定：不砍能力、不留任意频道的兼容层，只收紧渲染层输入与本机资源之间的信任边界。两个 `BrowserWindow` 改为 `contextIsolation: true` / `nodeIntegration: false` / `sandbox: true`，移除 `@electron/remote`（`package.json` 依赖与 `vite.config.ts` 的 external 一并删除）；`src/main/preload.ts` 只通过 `contextBridge` 暴露 19 个具名方法，频道全部取自 `IPC_CHANNELS`，不再有任意频道转发，invoke 错误直接向上传播；`src/global.ts` 去掉掩盖类型错误的索引签名，浏览器指纹修正拆到 `src/main/browser-fingerprint.ts`。渲染层 6 个文件里直接使用的 `ipcRenderer` 与 `@electron/remote` 全部迁到具名方法。新增 `src/main/path-guard.ts`：目录须由主进程原生 `dialog` 显式授权，`READ_FILE_IN_DIRECTORY` / `MERGE_VIDEO` 的每个路径都须落在授权目录内（含 `realpath` 防符号链接与前缀欺骗），其余 handler 逐个校验参数；`OPEN_EXTERNAL` 只放行 `EXTERNAL_TARGETS` 枚举里的 https 目标。顺带修复 3 处既有缺陷：`Files2video.tsx` 的 `mergeVideo` 缺 `outputPath`、`VoiceTools.tsx` 用 `send` 调用只注册了 `handle` 的字幕频道、`Settings.tsx` 4 个按钮调用的 `requestOpen*` 方法原本根本不存在（现改为主进程 URL 白名单打开官方站点）。新增 3 组单测共 24 例；`npx tsc --noEmit` exit 0、`npx vitest run` 135/135 通过、`npx vite build` 成功（preload 产物只 `require("electron")`）；用两个隐藏 `BrowserWindow` 做受控冒烟得到 `SMOKE_OK`：`window.desktop` 正常注入、19 个具名方法齐全、通用 `send` 与 `require` 均为 `undefined`。验证边界：冒烟加载的是空白页而非真实业务 UI，完整交互回归与 CI 无头冒烟仍属「可复现安装」任务。本机 `ELECTRON_RUN_AS_NODE=1` 会让 Electron 以纯 Node 模式运行而不提供内置模块，冒烟时需去掉该变量。提交：`c10ba499`。

- [x] 把 meta-note 独有的可合并功能迁入 `apps/digital-twin/system-management`，使 digital-twin 成为功能完整的主项目。按 2026-11-13 的决定：不开分支、直接在 master 处理，互斥或会破坏安全基线的部分先跳过。迁入 13 个源码文件（笔记生成 `note.controller` / `note.service`、学习机队列 `study.controller` / `study.service`、7 个下载与转写 provider、`VideoNote.tsx`），补 `youtube-dl-exec` / `kafkajs` / `ioredis` 三个依赖、`STUDY_CONSTANTS` 与 5 条路由常量、合并后的 `.env.example`、`VITE_WEB_SERVER_PORT` / `VITE_VOICE_API_URL` 的构建注入；在 `app.module.ts` 注册 9 个 provider、`desktop-main.ts` 与 `http-server.ts` 接通 5 条路由，`App.tsx` 增加「视频笔记」第三入口。三处必要适配：`NoteController` 原本用 Nest 装饰器注入参数，但本项目路由是 Express 手动绑定，故改成 `(req, res)` 处理器并保持原响应契约；`Transcriber` 在源项目是固定返回 placeholder 文本的占位实现，改为真实调用 `WHISPER_URL` 的 Whisper 兼容端点并在无结果 / 传输失败时抛错；删掉 `note.service.ts` 里未使用的 `NoteController` 循环 import。新增 16 个行为断言单测（4 个 spec 文件），含通过真实 Express 端口发起 `fetch` 验证新路由已注册。验证：`npx tsc --noEmit` exit 0、`npx vitest run` 151/151 通过（原 135 + 新 16）、`npx vite build` 成功（preload 仍为 3 模块 / 4.61 kB，隔离基线未变）、`git diff --check` 干净、lefthook 三项守卫全绿。跳过的部分与仍需人工决策的事项已登记为上面两条 `[ ]` 任务：两套互斥的 `LLMController`、ChatGPT / DeepSeek 监控窗口与注入脚本、`AuthController` 的 token 写入、meta-note 的两个子模块 / 文档 / Rust DDD 空壳 / 两个无调用方的类型，以及仓库腾空、品牌与 `APP_PROTOCOL` 命名。运行边界：Kafka / Redis / Whisper / yt-dlp 都是外部依赖，本次只验证编译、单测与 HTTP 路由，未做端到端实跑；本地 HTTP 服务仍是 `CORS *` 且监听所有网卡，新增的写操作路由扩大了无鉴权面，已在「为本地 HTTP 服务增加鉴权」任务里补记。提交：`10b4b4e8`。

- [x] 保留并迁入 meta-note 的 ChatGPT / DeepSeek 监控窗口，关闭「合并监控窗口与两套 LLM 控制器」任务。按 2026-11-13 用户决定：监控窗口保留，按 `payload.model` 分流。迁入 `desktop-chatgpt.ts` / `desktop-deepseek.ts` / `browser/{base,enhanced}-browser-window.ts` / 3 个 `main/injects/*` / 2 个会话数据类型，并新增 `monitor-preload.ts` 与 `monitor-llm.controller.ts`。两个监控窗口统一到 `contextIsolation: true` / `nodeIntegration: false` / `sandbox: true`，使用独立 `persist:chatgpt` / `persist:deepseek` 分区；注入脚本不再依赖已删除的 `window.desktop.send` 与 WebSocket 兜底，只通过 `window.monitor` 的 `reportSseChunk` / `requestOpenExternalLogin` 两个具名方法上报与跳转；SSE 分片改由 `MONITOR_SSE_CHUNK` 频道进入主进程 eventBus，并修正源项目监听 `sse_chunk` 与发射 `sse-chunk` 不一致导致转发失效的缺陷。`/v1/chat/completions` 保持单一路由：`local`（默认）走原 `LLMService` / `LOCAL_LLM_PROVIDER`，`chatgpt` / `deepseek` 委托 `MonitorLLMController`，`getTags` 同时列出三个模型；`AuthController` 接回真实 session token 写入（ChatGPT 写分区 cookie、DeepSeek 写可配置 cookie），且不再把 token 打到日志。渲染层 `Settings.tsx` 改为通过具名方法打开应用内监控窗口；`AudioMonitor.tsx` 里残留的任意频道 `desktop.send` 一并改为具名音频方法。验证边界：按用户「不要做任何测试」的要求，本轮未运行 vitest / e2e / 构建，仅做 `npx tsc --noEmit` 静态类型检查（exit 0）、`git diff --check` 与 lefthook 三项守卫；窗口冒烟与单测按用户决定豁免。提交：`2a318bd1`。

- [x] 按 2026-11-13 的 6 条决定关闭仓库与品牌决策：原本的 meta-note 仓库不做任何改动（不打归档 tag、不清空 `main`、不搬 `browser-extension` / 文档 / Rust DDD 空壳），digital-twin 继续作为 monorepo 内的集成项目，根 workflow / `DEVELOPMENT.md` / 目录结构均不调整；用户可见名称统一为 `meta-note元笔记`（`package.json` 的 `productName`、`index.html`、两份 README、主窗口与两个监控窗口标题、`Settings.tsx` 标题与欢迎语），`APP_PROTOCOL` 继续用 `meta-note` 以避免破坏已注册 URL scheme 与深度链接。lockfile 确认为生成文件、不入库，已同步改写「可复现安装与 Electron 运行时冒烟检查」任务的建议与关闭条件，去掉 `npm ci` 路线；该任务因用户明确本轮不做测试而继续保持 `[ ]`。提交：`2a318bd1`。

- [x] 关闭「为 digital-twin 本地 HTTP 服务增加鉴权并绑定到回环地址」。`HttpServer` 改为监听 `127.0.0.1`，去掉 `Access-Control-Allow-Origin: *`，改为只放行 `file://`（`Origin: null`）、Vite dev server 与自身端口的显式来源白名单，并对私有网络预检回 `Access-Control-Allow-Private-Network`；所有路由统一要求逐会话 token（`Authorization: Bearer` / `X-Local-Api-Token` / SSE 场景的 `?token=`），用 `timingSafeEqual` 常量时间比较，未设置 `LOCAL_API_TOKEN` 时每次启动随机生成。渲染层通过新增的具名桥接 `getLocalApiToken` 取值，新增 `src/renderer/services/local-api.ts` 统一为本地 fetch 附加 token，`VideoNote.tsx` / `VoiceTools.tsx` / `SubtitlesOverlay.tsx` 全部改走该 helper，`/api/tts/download` 的 EventSource 改用带 token 的 URL；外部 `VITE_VOICE_API_URL` 不会收到 token。`SystemController` 对 `filePath` / `videos` / `outputPath` 做类型、长度、空字节与 `..` 路径校验；README 与 `.env.example` 补充鉴权及 `DEEPSEEK_SESSION_COOKIE_NAME` 说明，并让 `http-server.spec.ts` / `preload.spec.ts` 的断言与新契约保持一致（未运行测试）。提交：`2a318bd1`。

- [x] 澄清 5051 本地 HTTP 服务的「外部调用方」问题，撤销上一轮结论里的「外部适配」待办。2026-11-15 核实：该服务由 Electron 主进程内嵌、绑定 `127.0.0.1`，monorepo 内唯一消费方是自身渲染层（`src/renderer/services/local-api.ts` 的 `apiFetch`）；`AudioContext.tsx` 的 3 处 fetch 指向 `localhost:8000` 语音服务，`services/api/alertRule.ts` 指向 `localhost:8082` 数字孪生后端，都不属于该鉴权面。`llm.controller.ts` / `remote-llm-provider.ts` 里的 `/v1/chat/completions` 是出站调用本地 LLM provider，也不是调用方。「外部数字孪生程序需要适配鉴权」沿用自早期 TODO 的推测性措辞，代码里没有对应入口，故撤销。原 `apps/meta-note/browser-extension`（按决定不动）里只有一个 `studyBaseUrl` 设置项默认填 `http://localhost:5051`，但 `studyAutomationBaseUrl` 仅在 `popup/App.tsx` 里读写 `chrome.storage.local`，全仓无发请求处，`public/manifest.json` 的 `host_permissions` 也不含 localhost，因此不构成调用方。已把 `system-management/README.md` 的鉴权段改为明确「internal API，monorepo 内无其他调用方」并删去 `External callers` 措辞；`.env.example` 的 `LOCAL_API_TOKEN` 说明本就中性，未改。提交：`f9090515`。

- [x] 修完「重新启用 `apps/client` e2e 套件」任务里所有本机可闭环的代码缺陷。按 Detox 20 重写 `e2e/detox.config.js`（`apps` / `devices` / `configurations`，默认 Android 模拟器 + 可选真机，删掉 `composition` / `session` 与 `油脂` 字段），`e2e/jest.config.js` 改用 `detox/runners/jest/*` 的 globalSetup / testEnvironment / reporter，`e2e/setup.ts` 移除未声明的 `detox-adapter-jest` / `detox-matchers`，`e2e/app.spec.ts` 修正 `../pages` 导入并替换 4 类旧设备 API，`package.json` 三个 e2e 脚本显式指定 `--config-path` 与 `-c`。静态验证全部通过，但按 2026-11-13 的决定未运行任何测试；该任务仍需真机 / 模拟器跑通一条真实用例，故保持 `[ ]`。提交：`f9090515`。

- [x] 按 2026-11-15 第 2 条决定把数字孪生状态获取改为事件驱动。`renderer/hooks/useDigitalTwinData.ts` 删除 `POLL_INTERVAL_MS = 15000` 与 `setInterval`，`refreshAll()` 保留为全量对账，只在首屏、WS 重连与手动刷新时调用（`needsReconcileRef` 在 `onClose` 置位、`onOpen` 触发一次对账，首次连接不重复）；新增 `refreshAlerts()` / `refreshStats()` 两个局部刷新，`ALERT_CREATED` 只回查告警、`PRODUCTION_OUTPUT_UPDATED` 只追加产量曲线并回查统计，`DEVICE_STATUS_CHANGED` / `DEVICE_POSITION_UPDATED` / `AGV_POSITION_UPDATED` 继续走 `patchDeviceFromEvent` 增量 patch；`acknowledgeAlert` / `resolveAlert` 改为局部刷新告警，不再全量拉取；`devicesRef` 保存设备列表供告警名称映射使用；错误文案改为英文以符合代码原则。服务端事件 payload 仍不完整，已另立 `[ ]` 任务。静态验证：`npx tsc --noEmit` exit 0、`git diff --check` 干净；按决定未运行任何测试。提交：`15a4a873`。

- [x] 按 2026-11-15 第 1、3 条决定删除 `docker-compose.digital-twin.yml` 的 `digital-twin-frontend` 服务段。该容器是 `5b6f2847 ECharts integrate into digital twin.` 引入的 Web 版历史遗留，构建的是 nginx 静态托管产物，与纯 Electron 的本地工厂单档形态冲突，且其 `VITE_*` 运行时注入对 Vite 构建期变量本就不生效。删除后用 PyYAML 解析确认 compose 只剩 `factory-service` 一个服务。`k8s/digital-twin/` 四个 yaml 与 nginx 资产保留，清理动作登记为 `[ ]` 任务待确认。提交：`15a4a873`。

- [x] 按 2026-11-15 的 7 条决定关闭「提供分级部署档位并明确产品定位」任务：明确不做分级部署、不引入云端方案，本项目完全面向本地工厂；交付形态定为纯 Electron；模拟／演示数据只允许出现在测试脚本或初始化演示；现场协议做通用版本；时序库必须实际接入；对接 `server/` 的接口通讯必须经 zk + gateway 鉴权，Electron 内部消化的接口不验证。同步把 9 条架构任务按新口径改写，并写入 gateway / zk 鉴权链路的核实结论与 Apache PLC4X 的标准库调研结论。按决定未运行任何测试。

- [x] 按 2026-11-15 追加的决定禁止绕开网关直连 service，并直接落地改造。渲染层新增 `renderer/config/gateway.ts` 作为唯一基址，`config/digital-twin.ts` / `config/mes.ts` / `services/api/alertRule.ts` / 两个生成客户端统一到 `gateway:10081/factory-service/**`，删除 `10101` / `8082` / `8080` 三组直连端口与 `VITE_DIGITAL_TWIN_*` / `VITE_MES_*` / `VITE_API_URL` 注入；新增 `services/gateway-client.ts`（`gatewayFetch` / `installGatewayAuth` / `buildGatewayWsUrl`）与具名桥接 `getGatewayToken`，JWT 由主进程读 `.env` 的 `GATEWAY_TOKEN`，不进渲染层 bundle。网关侧增加 `lb:ws://factory-service` 显式 WS 路由与 `StripPrefix=1`，移除 `/ws/digital-twin` 与 `/ws/**` 的免鉴权排除，`UserAuthFilter` 支持 WS 握手 `?token=`（校验后提升为 Authorization 头、转发前剥离 query 参数）；服务侧 `DigitalTwinAuthHandshakeInterceptor` 改为校验 `X-Internal-Token` 后才采信网关注入的用户头。`docker-compose.digital-twin.yml` 改为 `ports: []` 并补 zk 连接串与可覆盖的 kafka 地址，两份 README 与 `.env.example` 写明边界。Electron 内部 `5051` / `5050` 保持原有会话 token，不引入 JWT。同步澄清 PLC4X 1.0.0「API 模块 Java 17、驱动 / SPI / 工具 Java 21」的含义与官方依据。按决定未运行任何测试，只做 `npx tsc --noEmit`、`git diff --check` 与 YAML 静态解析；剩余 token 签发、REST 内部令牌注册与运行验证另立任务。
