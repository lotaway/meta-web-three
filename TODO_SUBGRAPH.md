# 基于 Graph Node（subgraph）的 EVM 事件同步与区块记录

> 目标：引入 `lotaway/graph-node` fork，用 subgraph 事件配置实现 EVM 链的区块同步、canonical chain 校验与原始事件记录，替代自建 RPC 同步引擎；业务投影、幂等、下游通知仍自建。Solana 为次要，保持自建轻量同步。与 `TODO_BLOCK_REORG.md` 配套：该文档的 Phase 2 与 Phase 3.1-3.4 由 Graph Node 承担，其余任务移交本文档。

## 现状核对结论（2026-08 调研）

调研范围：`~/Downloads/graph-node`（fork 源码）、`docker/`、`docs/environment-variables.md`、`LOTAWAY_CHANGES.md`。

1. `lotaway/graph-node` 是 `graphprotocol/graph-node` 的 fork（Rust），定制内容仅为国内可用性：TUNA/USTC/Go proxy 镜像、docker-compose env 编排、多 EVM RPC、区块数据存储压缩（约 -40%）。
2. **链支持仅 `chain/ethereum` + `chain/near`，无 Solana**；项目主链为 EVM，Solana 次要，故引入成立。
3. Graph Node 原生内置：区块同步、父哈希校验、共同祖先查找、reorg 检测/回滚/重放、幂等（block_hash 维度）、`ETHEREUM_REORG_THRESHOLD` 配置、GraphQL 查询。**对应 `TODO_BLOCK_REORG.md` Phase 2 全部与 Phase 3.1-3.4，无需再自建。**
4. 部署依赖：graph-node 二进制/镜像 + IPFS（`ipfs/kubo`）+ 独立 PG 库（需扩展 `pg_trgm`/`btree_gist`/`postgres_fdw` + `GRANT USAGE ... postgres_fdw`）。fork 默认 `postgres:14`，需与仓库统一为 `postgres:16`。
5. **不覆盖**：写业务库（Graph Node 只写 graph-node 库）、链上写交易、reorg 事件通知下游、业务投影回滚、确认门槛、查询层 block 元数据 —— 这些仍在 Java 侧自建（承接 TODO_BLOCK_REORG 对应任务）。
6. 服务端口：8000 GraphQL / 8020 admin（部署） / 8030 indexer；`_meta.block` 提供已索引高度。
7. **Solana 选型**（详见 Phase 5）：graph-node 上游 master 同样无 `chain/solana`，Solana subgraph 必须走 Firehose 数据源（`firehose-solana`，Apache-2.0 可 fork + validator/geyser 或托管 gRPC），自建成本高；Solana 为次要，**选定 Helius Webhooks（托管推送）**，仅做事件落库与确认门槛。

---

## Phase 1: 基础设施部署

- [ ] **docker-compose 增加 graph-node 编排**（参考 fork `docker/docker-compose.yml`）
  - 新增 `graph-node` 服务（fork `Dockerfile` 或构建产物），端口映射 8000/8020/8030
  - 新增 `graph-ipfs`（`ipfs/kubo:v0.17.0`），内网 5001
  - 新增 `graph-postgres`：**镜像改为 `postgres:16`**（与仓库统一），`-E UTF8 --locale=C`
  - 全部并入 `meta-web-three` docker network
- [ ] **初始化 graph-node 专用 PG 库与扩展**
  - 建库 `graph-node`；执行扩展 `pg_trgm`、`btree_gist`、`postgres_fdw`
  - `GRANT USAGE ON FOREIGN DATA WRAPPER postgres_fdw TO <graph_user>`
  - 验证 `cargo run`/镜像启动后建表成功
- [ ] **env 配置化（Compose 与 k8s 两套）**
  - 统一 `GRAPH_*` 环境变量：`postgres_host/user/pass/db`、`ipfs`、`ethereum`、`GRAPH_LOG`
  - k8s：新增 deployment + service + secret（graph 库口令），并入 `deploy-all.yaml`
  - `ethereum` 参数支持**多 RPC**：`"base:[capabilities]:url1,url2"`（fork start 脚本已支持）
- [ ] **确认 reorg 与最终性配置**
  - 设置 `ETHEREUM_REORG_THRESHOLD`（回滚深度，对应 TODO_BLOCK_REORG Phase 2.8/5.2）
  - 记录各链 finality：base 等 EVM 用确认数策略
- [ ] **启动冒烟验证**
  - 启动后访问 `:8030/graphql` 查询 `{ indexingStatuses { subgraph synced headBlockNumber } }`
  - 确认无扩展/连接报错，indexer 能推进 head

## Phase 2: subgraph 开发

- [ ] **建立 subgraph 工程目录**
  - 新建 `contracts/evm/subgraphs/`（或独立仓库），`package.json` 引入 `@graphprotocol/graph-cli`（pnpm）
  - 目录含 `subgraph.yaml`、`schema.graphql`、`src/mapping.ts`
- [ ] **导出合约 ABI 与事件清单**
  - 从 foundry artifacts（`contracts/evm/out/`）导出目标合约 ABI
  - 列全事件签名（如 payment 的 `CryptoTransfer`，promotion/合约其余事件）
- [ ] **编写 schema.graphql**
  - 实体：`ChainEvent`（`id` = `chainId-blockHash-txHash-logIndex`）、`CryptoTransfer`、`Order` 等
  - 每个实体字段含 `blockNumber`/`blockHash`/`transactionHash`/`logIndex`/`timestamp`
- [ ] **编写 subgraph.yaml**
  - `dataSources`：`kind: ethereum/contract`，`network: base`，目标 `address`/`startBlock`
  - `eventHandlers` 与 mapping 一一对应
- [ ] **编写 mapping.ts**
  - 每个 handler 用事件唯一键建实体并落库（Graph Node 自动维护 canonical/REVERTED，reorg 时自动回滚并重跑 handler）
  - 记录 `event.block.number/hash`、`event.transaction.hash`、`event.logIndex`
- [ ] **构建与部署**
  - `graph codegen && graph build` 通过
  - `graph deploy` 到本地节点（`--node http://localhost:8020`，`--ipfs`）
  - 验证 `indexingStatuses` 显示 `synced: true`，实体查询有数据

## Phase 3: Java 消费侧（业务库事件记录）

- [ ] **新建原始事件相关表（PG 方言）**
  - `chain_block`：`number`/`hash`/`parent_hash`/`timestamp`/`status(pending|confirmed)`（承接 TODO_BLOCK_REORG Phase 3.1）
  - `chain_raw_event`：`chain_id`/`block_number`/`block_hash`/`transaction_hash`/`log_index`/`payload`/`status`，**唯一索引 `(chain_id, block_hash, transaction_hash, log_index)`**（承接 Phase 1.5/3.2/3.3）
  - `chain_sync_cursor`：最后消费的 `block_number`/`block_hash`
- [ ] **GraphQL 客户端接入**
  - 新增 Java 客户端调用 `http://<graph-node>:8000/subgraphs/name/<name>`（复用 common 的 HTTP/RestTemplate 基建）
  - 查询 `_meta.block.number` + 实体分页查询
- [ ] **同步 Job（Quartz 复用 common 已配置集群调度）**
  - 轮询 GraphQL，`last_indexed > cursor` 时拉取增量事件
  - 事件批量落 `chain_raw_event`（按唯一键 upsert，幂等）
  - 失败重试、断点续跑（重启后从 cursor 继续）
- [ ] **事件状态机**
  - `chain_raw_event.status`：`PENDING → CONFIRMED → PROCESSED`；`REVERTED`（承接 Phase 3.3）
- [ ] **业务投影管线**
  - 投影表补 `source_event_id`/`block_number`/`block_hash`（承接 Phase 1.6）
  - `PROCESSED` 才触发投影写入；写路径以事件唯一键做幂等（承接 Phase 3.6）
- [ ] **管理/状态 API**
  - `GET /internal/chain/sync`：当前 cursor、graph-node head、同步延迟、最近 reorg（承接 Phase 4.3）
  - `GET /internal/chain/reorgs`：最近回滚记录

## Phase 4: 回滚、通知与确认门槛

- [ ] **reorg 检测接入**
  - 对比 `chain_sync_cursor` 与 graph-node 新 head，发现回退/换链时标记 `chain_raw_event.status=REVERTED`（承接 Phase 3.4）
  - 以 `_meta` 的 reorg 信号或 block hash 变化为触发源
- [ ] **reorg 领域事件发布（Kafka/event-sdk）**
  - 新增 `BlockchainReorgEvent`（extends `BaseEvent`），扩展 `EventType`
  - 消息含 `chain_id`/`common_ancestor`/`old_head`/`new_head`/`reorg_depth`（承接 Phase 4.1/4.2）
- [ ] **业务投影回滚**
  - 收到 reorg 事件后按 `block_number > common_ancestor` 撤销/重打投影（承接 Phase 3.5）
  - 累加统计禁止不可逆 `+/-`，改为按 canonical 事件重算（承接 Phase 3.6/5.1）
- [ ] **确认门槛（不可逆外部副作用）**
  - `payment-service`：`processCryptoTransfer` 增加确认数等待（finalized/确认数）后再置 `COMPLETED`（承接 Phase 4.6/5.1）
- [ ] **outbox/inbox 一致性**
  - "事件状态变化 + 投影变化"同库同事务，或引入 outbox（承接 Phase 3.8）

## Phase 5: Solana（次要）选型与确认链路

> 调研结论（2026-08）：graph-node（含上游 master）仓库无 `chain/solana` 模块，Solana subgraph 必须走 Firehose 数据源——`firehose-solana`（streamingfast，Apache-2.0，可 fork）+ validator/geyser 插件或托管 gRPC，自建成本高。Solana 为次要业务，**选型定为 Helius Webhooks（托管推送）**，不引入自建索引。

- [ ] **选型确认（已定：Helius Webhooks）**
  - 托管推送模式：Helius 订阅链上事件，按配置匹配后 POST 解析好的 JSON 到业务 HTTP 回调，无需自跑 validator/firehose-solana
  - 提供公网回调接口（如 `/internal/chain/solana/webhook`）
  - 配置：`account_addresses`/`transaction_types`/Enhanced 解析；事件自带 `signature`/`slot`/确认状态
  - 计费：1 credit/event，webhook 配置操作 100 credits/次
- [ ] **备选方案留档（不实施）**
  - 双重 fork：graph-node fork（EVM）+ `firehose-solana` fork（SVM），即 The Graph 官方生产架构；需 validator + geyser 插件或托管 gRPC，graph-node 配置 `[chains.solana-*]` + `provider: [firehose]`，subgraph 用 `kind: solana`（account/transaction/block handler 模式，与 EVM 事件 handler 不同）
  - 自建轻量轮询：`SolanaRpcClient` 扩展 `getSignatureStatuses`/`getBlockHeight`（`commitment: "finalized"`）+ BlockCursor 轮询 Job，按 `tx_signature` 回查（承接 TODO Phase 2.4/5.3）
- [ ] **Solana 事件落库与确认门槛**
  - 回调事件按 `signature` 幂等 upsert，回填 `tb_solana_activity`/`tb_solana_listing`，补 `source_event_id`/`block_number`/`block_hash`（承接 Phase 1.6）
  - 未达确认条件不置不可逆状态（承接 Phase 5.1/5.3）
- [ ] **配置化**
  - `blockchain.solana.webhook.*`：回调路径、签名/密钥校验、`confirmation-blocks`

## Phase 6: 测试与故障演练

- [ ] **本地链编排**
  - docker-compose 增加 `anvil`（或 fork base）作为 graph-node RPC
  - 配置 reorg 演练用脚本（mine 分支区块 → 重组）
- [ ] **单区块/多区块 reorg 演练**
  - 验证 Graph Node 自动回滚 + 重放，Java 侧 `chain_raw_event` 状态正确翻转（承接 Phase 6.1/6.2）
- [ ] **e2e 测试**
  - 新区块 → 事件入库 → reorg → 回滚 → 新链重放 → 查询结果一致（承接 Phase 6.6）
- [ ] **幂等测试**
  - 重复投递、消费者重启、RPC 切换场景不产生重复实体/重复统计（承接 Phase 6.4/6.5）
- [ ] **一致性巡检脚本**
  - SQL 反查投影表 `source_event_id`/`block_hash` 是否引用非 canonical/REVERTED 事件（承接 Phase 6.7）
- [ ] **告警与 dashboard**
  - 指标：同步延迟、head、回滚次数/深度、重放耗时（承接 Phase 6.8/5.4）

## Phase 7: 交付验收标准

- [ ] EVM 业务事件可追溯到 `chain_id + block_hash + transaction_hash + log_index`
  - 依赖 Phase 2（subgraph 落唯一键）+ Phase 3（业务库 `chain_raw_event`）
- [ ] 区块重组时业务数据不再写入错误链，且被替换链事件标记 `REVERTED`
  - 依赖 Phase 4（reorg 检测 + 投影回滚）
- [ ] 回滚后新链事件自动重放，不重复创建实体/统计/消息
  - 依赖 Phase 3（幂等 upsert）+ Phase 4（outbox/状态机）
- [ ] 服务重启后从持久化 cursor 继续同步
  - 依赖 Phase 3（`chain_sync_cursor`）
- [ ] 下游可通过状态 API 或领域事件明确知道哪些区块已回滚
  - 依赖 Phase 3/4（管理 API + Kafka 事件）
- [ ] Solana 次要链路至少有确认数门槛，不乐观置不可逆状态
  - 依赖 Phase 5
