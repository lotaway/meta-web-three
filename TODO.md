# TODO

## 永久性规范（持续约束）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES)

## 可执行任务

### 1. 需人工决策

- [x] 清理 common 模块剩余的扫描遗留，统一到自动装配。
- [x] 过滤并迁移非网关服务间调用，为 internal-token 全量开启铺路。
- [x] 把 `apps/client` 与 `apps/backstage-admin` 的测试覆盖补齐。
- [x] 清理两个客户端里的代码原则违规项。
- [x] 补齐两个客户端的响应式与可访问性缺口。
- [x] 让 `apps/backstage-admin` 的 API 客户端与 DEVELOPMENT.md 第 3 节保持一致。
- [x] 让 DEVELOPMENT.md 第 3 节与实际的 env 和生成器契约对齐。
- [x] 统一本地工厂单档部署的端口、环境变量与 compose 口径。

### 2. 需环境 / 权限

- [x] 重新启用 `apps/client` 的 e2e 套件并修复其配置。

### 3. 需决策 + 环境

- [x] 修复 `apps/client` 的鉴权 token 存储。
- [x] 替换 `apps/client` 里用抛异常代替实现的三个鉴权流程。
- [x] 落地 FRONTEND_PRICEPLES 的视觉与五维质量门禁。
- [x] 撤销已泄露的 classic PAT 并从 git 历史中清除。

- [ ] 把性能流水线指向真实的 staging 目标。
  - 问题：`GET /repos/lotaway/meta-web-three/actions/variables` 返回 `total_count: 0`，因此 `vars.PERF_BASE_URL` 未设置，每次运行测的都是内置的 `127.0.0.1:18087` mock；从未触达真实服务。历史 175 次 jmeter artifact 全部打 mock，项目从未有过真实 staging。
  - 建议：把 `PERF_BASE_URL` 仓库 / 环境变量设为可达的 staging 网关，重跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL 且 `use_mock=false`。已提供 `scripts/set-perf-base-url.ps1 -Url <real-url> -Dispatch` 一键完成设置 + 触发 + 验证。
  - 阻塞：CI 跑在 GitHub `ubuntu-24.04` runner 上，无法访问本机/内网地址；项目从未部署 runner 可达的 staging 网关。没有真实 URL 前无法关闭，编造 URL 只会让 CI 压测失败并污染报告。流水线保持 mock 明示标注，不产生虚假数据。
  - 环境要求：需要从 `ubuntu-24.04` runner 可达的 staging 网关、设置 `PERF_BASE_URL` 所需的仓库 / 环境变量写权限，以及 `workflow_dispatch` 权限。
  - 关闭条件：设置变量后跑一次 `workflow_dispatch` 和一次定时运行，确认步骤输出显示真实 URL、`use_mock=false`，且没有 MOCK DATA 标注。

- [ ] 按「演示数据仅用于初始化／测试」的边界重建设备数据通路。
  - 问题：`DigitalTwinSimulator.java` 标 `@Profile("dev")` 且用 `@Scheduled(initialDelay = 5000, fixedDelay = 10000)` 随机改设备状态；`DigitalTwinDevDataInitializer.java` 标 `@Profile("dev")` 注册 8 台演示设备；`MqttConfig.java` 默认 `mqtt.enabled:false`。而 `docker-compose.digital-twin.yml` 设 `SPRING_PROFILES_ACTIVE=prod`，覆盖了 `application.yml:7` 的默认 `dev` —— 容器启动后既没有演示数据也没有采集数据，`/api/digital-twin/devices` 为空。演示／模拟数据只能出现在测试脚本或初始化演示，不能因此放弃真实数据读取与接入。
  - 建议：把模拟器与 8 台演示设备从 `dev` 抽到显式的 `demo` 初始化器（或移进 `src/test` 的 fixture），仅在用户选择「演示模式」时写入一次，不再定时改写运行时状态；`prod` 必须接到下面「协议无关的通用现场接入层」的真实数据源；MQTT 开关由 compose 显式注入并在 prod 打开。状态实体补 `source`（REAL / DEMO）/ `quality` / `timestamp`，前端在演示模式下明确标注数据非真实。
  - 口径：假数据只能出现在初始化脚本或演示功能里（便于用户测试），除此之外所有通路按真实 / 通用 / 标准 / 兼容方向实现；实际用户未定、不可能有具体设备，不强求提供具体真实设备。`demo` profile 显式触发演示初始化，prod 只走真实数据源。现场数据解析配置化（`mes.telemetry.*`：timestamp/quality/unit 字段、设备码提取段、topic 过滤、质量映射、点位别名，支持 ISA-95 UNS）。
  - 环境要求：prod 通路的运行验证需要 MQTT broker（或任意 TelemetrySource 适配器）；标准接口与归一化逻辑本身可离线用单测验证。
  - 关闭条件：prod 能从真实数据源读到设备与遥测，`demo` 只做一次性初始化、不参与运行时状态变更，测试脚本里的模拟数据可复现，前端能区分 REAL / DEMO 与数据新鲜度。

- [x] 建设协议无关的通用现场接入层（当前没有真实工厂，目标是后续最小修改）。
- [x] 实际接入 InfluxDB 时序库，不允许留空依赖。
- [x] 明确「数字影子 / 闭环孪生」边界，并实现受控写回（如需）。
- [x] 验证 zk + gateway + 数字孪生专用账号 token 的整条链路。
- [x] 补齐服务端事件 payload 并让前端无需回查。
