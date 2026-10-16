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

GitHub Dependabot 告警共 53 条（High 20 / Medium 25 / Low 8）。后端 Maven 补丁已在本次处理（postgresql、minio、graphql-java、rhino、clickhouse-jdbc 版本升级）；以下为**尚未处理**的前端 npm 依赖告警，需评估升级兼容性后单独处理。

- [ ] **electron**（37 条，High 13 / Medium 20 / Low 4）：`apps/digital-twin/system-management/package.json` 当前固定为 `"33"`，远落后于修复版本。多数问题与沙箱/弹窗继承、跨源读取、Squirrel.Mac 更新竞态相关。需升级到 ≥ `41.10.6`（或至少 `39.8.10`），并回归验证桌面端功能与自动更新流程。
- [ ] **tinymce**（5 条，High 4 / Medium 1）：`apps/backstage-admin/package.json` 当前 `^6.8.6`。修复版本 `7.9.3`（部分 `7.1.0` / `7.0.0`），涉及 XSS 类问题；升级到 7.x 需回归富文本编辑相关功能。
- [ ] **vite**（3 条，High 1 / Medium 2）：`apps/digital-twin/system-management`（`^5.0.8`）与 `apps/backstage-admin`（`^7.2.4`）。修复版本 `6.4.3` / `6.4.2`，涉及 dev server 相关漏洞；注意 5.x 仍在受影响范围，需整体规划升级。
- [ ] **turbo**（2 条，Medium 1 / Low 1）：`apps/client/turbo-module/appsdk/package.json` 当前 `^1.10.7`。修复版本 `2.9.14`，属跨大版本升级（1.x → 2.x），需验证构建流水线兼容性。

## 运行环境问题（logs/run-server 近期日志）

- [ ] **gateway**：DNS 解析失败（`GUNDAM.mshome.net` NXDOMAIN）。检查 `hosts` 配置或移除该主机名相关引用。
- [ ] **factory-service**：PostgreSQL 连接失败（`org.postgresql.core.v3.QueryExecutorImpl.receiveErrorResponse`）。确认本地 PostgreSQL 是否启动、连接串及账号密码配置是否正确。
- [ ] **mall-trade-service**：价格引擎定时任务获取 BTC-USD 价格失败（`PriceEngineServiceImpl`）。需完善降级策略、重试机制或本地 mock 配置，避免定时任务持续报错。

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

