# 区块链事件同步与区块重组（Reorg）处理

> 目标：参考 Graph Node 的处理方式，使 `meta-web-three` 的区块链事件同步具备 canonical chain 校验、可回滚、可重放和下游一致性保障能力。所有链上事件写入业务库前，必须能够定位到 `block_number + block_hash + transaction_hash + log_index`，不能只依赖区块高度或交易哈希。

## 现状核对结论（2026-08 代码审计）

审计范围：`server/`（Java 后端全部领域）、`protos/`、`contracts/`、`shared/event-sdk`、`rocketmq/`、`infra/kafka`。

1. **整个仓库目前没有任何链上事件同步/索引/订阅代码**：全 `server/` 无 `eth_getLogs`、`eth_subscribe`、`EthFilter`、`getPastEvents`、Solana `getSignaturesForAddress`/`subscribe`。所有链上交互都是“请求驱动 + 乐观写”：REST 请求 → 签名 → `sendTransaction` → 把返回的 `tx_signature` 立即写库，**从不校验、从不确认、从不回读**。
2. 唯一的链上数据链接是 `tx_signature`（Solana 4 张表）+ `Exchange_Orders.crypto_transaction_hash`（payment），且都不带 block/slot 元数据。
3. `SolanaRpcClient.getTransaction()`（`getTransaction` L52-54）和 `getSlot()`（L57-60）是**死代码**——定义存在但全仓库无调用方。
4. 全库无 `block`/`event`/`cursor`/`canonical`/`reorg` 相关表；无 outbox/inbox。~~DB 引擎存在矛盾：`wallet-service/db/schema.sql` 是 **MySQL 方言**（`AUTO_INCREMENT`/`InnoDB`/`utf8mb4`），而 common 数据源是 **PostgreSQL**（`application-common-dev.yml:3`）——新表 DDL 前必须先统一目标引擎。~~ ✅ 已统一为 **PostgreSQL**：所有 MySQL 方言 DDL（wallet-service、hrm、project、invoice、settlement、developer-portal、inventory、common 等 9 文件）已转 PG 风格，k8s/Compose 镜像与配置（deploy-all.yaml、app-config.yaml、database-secret 键、storage PV、QUICKSTART/MAPPING/README）均已从 MySQL 切到 PostgreSQL（5432，`postgres:16-alpine`），服务端 MyBatis-Plus 分页方言同步改为 `DbType.POSTGRE_SQL`。新表 DDL 一律使用 PG 方言。
5. 事件总线事实标准是 **Kafka**（`shared/event-sdk` + 各服务 `@KafkaListener`）；RocketMQ 客户端类基本为死代码（`MQProducer` 无人 `start()`，`MQConsumer` 无调用方）；RabbitMQ 只有 k8s manifest 无 Java 接入。
6. `protos/blockchain/` 目录不存在；唯一 web3 proto 是 `protos/shared/web3/token.proto`（3 字段 `Token`）。Makefile 的 `PROTO_FILES := $(wildcard protos/*/*/*.proto)` 会自动纳入新增子目录，Maven 生成输出到 `server/common/src/main/java/com/metawebthree/common/generated/rpc/`。
7. 配置缺失：`blockchain.solana.rpc-url`（在用）与 `blockchain.evm.rpc-url`（占位，无 Java 读取）都只支持单 endpoint；无 commitment/finality/确认数/回滚深度配置；Solana RPC 调用未携带 `commitment`。
8. 调度基础设施（Quartz JDBC，clustered）已在 common 配置好（`application-common.yml`），但没有任何 Job 类，可作为 BlockCursor 调度基础复用。

结论：本改造**几乎从头新建**链上同步引擎，存量代码可复用的只有：RPC 编解码骨架（`SolanaRpcClient.call`）、事件总线（event-sdk/Kafka）、Quartz 调度、以及 Web3j 密码学工具（EIP-712/191）。业务写路径（`Solana*Service`、`ExchangeOrderServiceImpl`）是唯一需要“接入门禁”的既有代码。

---

## Phase 1: 现状盘点与统一数据模型

- [ ] **梳理 `server/`、`protos/`、`contracts/`、消息队列和各业务服务中所有链上事件入口、订阅方式、落库表和派生统计表**
  - 现状（盘点基本完成，可留档）：无任何事件订阅/扫描。链上入口仅两处：
    - Solana 提交链路：`server/blockchain-domain/wallet-service/.../infrastructure/solana/SolanaContractClient.java`（`buildAndSend` L153-250）、`SolanaRpcClient.java:46`（`sendTransaction`）
    - EVM：无本地链访问，仅 `payment-service/.../CryptoWalletServiceImpl.java`（`verifyTransaction` L248-309 只看状态字符串）与内存 stub `promotion-service/.../infrastructure/gateway/BlockchainServiceStubImpl.java`
  - 落库表：`wallet-service/src/main/resources/db/schema.sql` → `tb_wallet`、`tb_solana_listing`、`tb_solana_activity`、`tb_solana_commission_relation`、`tb_solana_keypair`；`traceability-service` → `product_info`/`trace_record`/`trace_event`（DDL 不在仓库）；`payment-service` → `Exchange_Orders`（`crypto_transaction_hash`）
  - 写库服务（乐观写）：`SolanaTokenService`、`SolanaActivityService`（`participate` L87-105 乐观 `participant_count+1`）、`SolanaMarketplaceService`（`buyGood` L91-116 乐观置 Sold）、`SolanaCommissionService`、`SolanaCouponService`
  - 派生统计：无链上派生统计；`platform-domain/commission-service` 是纯 DB 业务账本，与链无关
  - 改动范围：纯留档盘点，无需改代码，但结论需固化到后续任务的落库设计中

- [ ] **梳理每条链的同步模式：RPC/WebSocket、轮询、第三方索引服务或 Firehose，并记录各自是否支持 `removed`/`revert`/`fork` 信号**
  - 现状：Solana 仅 HTTP JSON-RPC 单 endpoint（`SolanaRpcClient.call` L69-85 丢弃 context/slot），无 WebSocket、无 `commitment`、无 `getBlock`/`getSignatureStatuses`/`getSignaturesForAddress`。EVM 无 RPC 客户端（`wallet-service/application.yml:10-11` 的 `blockchain.evm.rpc-url` 是占位，无 Java 读取）。所有 `removed`/revert/fork 信号均未处理
  - 改动范围：新增链读取能力与信号来源（无既有读取代码可改，属于新建）

- [ ] **统一定义链上事件元数据：`chain_id`、`contract_address`、`block_number`、`block_hash`、`parent_hash`、`transaction_hash`、`transaction_index`、`log_index`、`removed`、`observed_at`**
  - 现状：全仓库无以上任何字段。唯一链链接列：`tb_solana_listing/activity/commission_relation.tx_signature`、`Exchange_Orders.crypto_transaction_hash`（`payment-service/.../domain/model/ExchangeOrder.java:66`）。`tb_web3_user.chain_id`/`chain_type`（`user-service/src/main/resources/db/schema.sql:173-174`）是死列（`Web3UserDO` 无对应字段）。Solana 各表无 slot/blockhash/blockTime
  - 改动范围：新建统一元数据 proto/model（见下一步），并为存量表补 DDL 迁移列

- [ ] **在 `protos/blockchain/` 或共享模块中定义统一的 `BlockchainEvent`、`BlockRef`、`BlockReorg` 和同步进度消息**
  - 现状：`protos/blockchain/` 不存在。现有生成链路：Makefile `PROTO_FILES := $(wildcard protos/*/*/*.proto)`（自动纳入新目录）；Maven protobuf 插件（`server/common/pom.xml`）输出到 `server/common/src/main/java/com/metawebthree/common/generated/rpc/`；`make gen-java-dubbo` 生成 Java
  - 改动范围：新建 `protos/blockchain/chain.proto`、`protos/blockchain/reorg.proto`（`BlockchainEvent`/`BlockRef`/`BlockReorg`/`ChainCursor`/`ChainSyncStatus`）；Makefile 增加 `gen-blockchain` target；重新生成 Java 并提交 `server/common` 生成物

- [ ] **为事件增加唯一键：`chain_id + block_hash + transaction_hash + log_index`；禁止只用 `transaction_hash` 去重**
  - 现状：无有效去重键。`payment-service/.../infrastructure/persistence/mapper/ExchangeOrderRepository.java:46-47` `findByCryptoTransactionHash` 仅按 tx hash 去重（不满足要求）；Solana 各表仅 row PK + 常规索引；`traceability-service` 无幂等键（`TraceabilityCommandService.addTraceEvent` 重复调用重复插入）
  - 改动范围：链原始事件表新增唯一索引 `(chain_id, block_hash, transaction_hash, log_index)`；`ExchangeOrderRepository` 查询改为带 block 维度；既有写路径改为按唯一键 upsert

- [ ] **为所有链上派生数据补充来源事件引用和区块元数据，至少包括 `source_event_id`、`block_number`、`block_hash`**
  - 现状：所有派生表无这些列。`tb_solana_*` 只有 `tx_signature`；`tb_wallet.balance` 是本地 `DECIMAL`，从不链上同步；`tb_solana_activity.participant_count`（`SolanaActivityService.java:99-102`）与 `tb_solana_listing.status`（`SolanaMarketplaceService.java:105,129`）为乐观自增/置位，无来源事件引用
  - 改动范围：派生表迁移新增 3 列；`SolanaActivityService`/`SolanaMarketplaceService`/`SolanaCommissionService` 写路径回填；未来重放需要时可回读链上状态重算（PDA 可本地重推导）

## Phase 2: 区块同步与 canonical chain 校验

- [ ] **实现按区块同步的 `BlockCursor`，持久化最后处理区块的 `number`、`hash`、`parent_hash` 和同步状态**
  - 现状：无任何游标机制。`SolanaRpcClient.getSlot()`（L57-60）为死代码。Quartz JDBC 已配置（`server/common/src/main/resources/application-common.yml` L44-61，`auto-startup: true`，clustered），可为调度基础复用；但无任何 Job 类
  - 改动范围：新建 cursor 表 + 实体/mapper + 调度 `@Scheduled`/Quartz Job；扩展 `SolanaRpcClient`（新增 `getBlock`/`getBlockHeight`）；如含 EVM 需新建 Web3j 客户端。建议承载位置：wallet-service（已有 Solana/EVM 配置上下文）或新建 `blockchain-sync-service`

- [ ] **每处理新区块前校验：`new_block.parent_hash == cursor.block_hash`**
  - 现状：无区块读取能力（Solana 无 `getBlock*`，EVM 无 client），无父哈希字段
  - 改动范围：RPC 层新增 `getBlock`+`getBlockTime` 口径（Solana 需在 mock/测试 RPC 中稳定返回 `parentSlot`/`previousBlockhash`）；cursor 校验逻辑

- [ ] **发现父哈希不匹配时暂停业务事件提交，进入 reorg 检测流程，不得继续盲目消费新区块**
  - 现状：不存在“提交 gate”。业务写路径全部独立乐观写：`SolanaActivityService`、`SolanaMarketplaceService`、`SolanaCommissionService`、`SolanaTokenService`、`payment-service/.../ExchangeOrderServiceImpl.java:211-226`（`processCryptoTransfer`）
  - 改动范围：在以上业务写入口插入游标状态门禁（`HEALTHY` 才允许提交链上派生数据）；引入状态机（见下）并对外暴露

- [ ] **实现从 RPC 获取区块头的能力：按高度和 hash 查询区块，并支持配置多个 RPC endpoint**
  - 现状：`SolanaRpcClient` 构造器 `@Value("${blockchain.solana.rpc-url}")`（L20）单 endpoint；无按 hash/高度取区块方法；EVM 无 client
  - 改动范围：配置改 `blockchain.solana.rpc-endpoints: [...]` + failover/轮询（连接到 `SolanaRpcClient` 与新的 EVM client）；RPC 方法新增 `getBlockBySlot`/`getBlockByHash`、EVM `eth_getBlockByNumber`/`eth_getBlockByHash`

- [ ] **实现共同祖先查找：从当前 cursor 和新链头逐级回溯，找到最后一个双方 hash 相同的 canonical block**
  - 现状：无（全新逻辑）
  - 改动范围：新建 reorg 检测服务；保证幂等与按深度上限（见下）

- [ ] **增加同步状态机：`SYNCING`、`REORG_DETECTED`、`ROLLING_BACK`、`REPLAYING`、`HEALTHY`、`FAILED`**
  - 现状：无任何状态机；Solana 各服务是纯 `@Service` 无状态
  - 改动范围：新增 `ChainSyncState` 模型 + 持久化 + 状态迁移逻辑；`FAILED` 后停止提交

- [ ] **配置并记录确认策略：普通区块确认数、finalized/safe block（链支持时）、最大允许回滚深度**
  - 现状：无确认配置。Solana RPC 调用未携带 `commitment`（`sendTransaction` L46-49、`getLatestBlockhash` L63-66）；`blockchain.*` 仅 `rpc-url`/`chain-id`（`wallet-service/application.yml:9-13`）
  - 改动范围：`application.yml`（或 common 配置）新增 `blockchain.<chain>.confirmation-blocks`、`finalized`、`max-reorg-depth`、`reorg-threshold`；RPC 调用与查询带 commitment/finality 参数

- [ ] **对超过最大回滚深度的情况告警并暂停自动提交，避免错误数据继续扩散**
  - 现状：无告警/指标设施（仓库内无 Prometheus/Grafana 配置；common 只有统一日志与 RocksDB 记录）
  - 改动范围：状态机置 `FAILED` + 日志告警 + 管理 API 暴露状态；指标（见 Phase 5）

## Phase 3: 可回滚事件存储与业务数据回滚

- [ ] **建立 canonical block 表，保存区块号、区块 hash、父 hash、时间戳、状态和确认状态**
  - 现状：全 server 无 block 相关表（已枚举 200+ `@TableName`，无匹配）
  - 改动范围：新建 `chain_block` 表（含 pending/confirmed 状态）；DDL 引擎先与运维统一（见现状风险点 4）

- [ ] **建立原始链上事件表，保存完整事件元数据和原始 payload；原始事件与业务投影分离**
  - 现状：无任何原始事件持久化。Solana 交易回读有 `SolanaRpcClient.getTransaction`（L52-54，`jsonParsed`，可解析 `err`/`slot`/`blockTime`/`meta`）但从未调用；EVM 无读取
  - 改动范围：新建 `chain_raw_event` 表 + 写入服务（用 `getTransaction`/`eth_getLogs` 回溯）；表与业务投影解耦，采用“原始事件 → 投影”管线

- [ ] **为原始事件增加状态：`PENDING`、`CONFIRMED`、`REVERTED`、`PROCESSED`、`REPLAYED`**
  - 现状：无事件状态概念
  - 改动范围：`chain_raw_event.status` + 状态迁移；`PROCESSED` 表示投影已应用

- [ ] **实现按共同祖先区块回滚：将祖先之后的 block/event 标记为 `REVERTED`，或在明确业务要求时删除**
  - 现状：无
  - 改动范围：新建回滚服务；区块/事件表级联更新

- [ ] **实现业务投影回滚：删除或撤销受影响事件产生的实体、余额、订单、统计、库存和聚合结果**
  - 现状：投影仅存在于乐观写表：`tb_solana_listing`、`tb_solana_activity`、`tb_solana_commission_relation`（`Solana*Service` 写入）、`Exchange_Orders`（`ExchangeOrderServiceImpl.processCryptoTransfer` L211-226 置 `COMPLETED`）、`tb_wallet.balance`（本地额）。`traceability-service` 为纯业务 HTTP 写，除非接入链上事件否则不受 reorg 影响
  - 改动范围：投影表完成 Phase 1.6 来源引用后再按 `block_number > common_ancestor` 撤销/重打；实现撤销需把各乐观写（状态/计数）改为可逆或可重算

- [ ] **对累加型统计禁止只做不可逆 `+/-` 猜测；优先支持按 canonical 事件重放重新计算**
  - 现状：`participant_count` 乐观 `+1`（`SolanaActivityService.java:99-102`）即不可逆猜测的典型;`downline_count`/`level`（commission）本地计算
  - 改动范围：累加字段改为“由 canonical 事件推导”或回滚时按来源事件集合重算（不可逆 `+/-` 不做）

- [ ] **为动态数据、缓存、搜索索引、对象存储引用和异步任务建立回滚/补偿策略**
  - 现状：无链上缓存；Redis（common 配置）仅基础设施；RocksDB 是本地日志（`RocksDBManager`），非业务缓存；ClickHouse（`meta_web_analytics`，common YAML）只服务 data-pipeline，非链上
  - 改动范围：若未来有链上投影缓存，先以事件回滚时强制失效替代；文档化补偿矩阵

- [ ] **确保回滚和重放具备幂等性，可重复执行而不会产生重复数据或重复扣减**
  - 现状：无幂等。`traceability addTraceEvent` 重复调用重复插入（`TraceabilityCommandService.java`）；Solana 写表无业务唯一约束；`payment` 依赖 `findByCryptoTransactionHash`（不满足 block 维度）
  - 改动范围：以 Phase 1.5 唯一键为幂等键；重放执行器先查后写/状态守卫

- [ ] **使用事务或 outbox/inbox 机制保证“事件状态变化”和“业务投影变化”最终一致**
  - 现状：无 outbox/inbox（全仓库 0 实现）；事件总线是 Kafka。~~DB 方言矛盾（见现状风险点 4）：`wallet-service/db/schema.sql` MySQL 风格 vs common-dev PostgreSQL~~ ✅ 已统一为 PostgreSQL（见现状风险点 4）
  - 改动范围：引入 outbox 表（与业务投影同库同事务）或事务内事件状态+投影更新；统一 DDL 引擎与迁移脚本（现有 `init_db.sh`/Postgres）

## Phase 4: 事件发布、下游通知与重放

- [ ] **设计并实现 reorg 领域事件：`BlockReorgDetected`、`BlocksReverted`、`EventsReverted`、`ReplayStarted`、`ReplayCompleted`**
  - 现状：无 reorg 事件。事件基础设施：`shared/event-sdk`（`BaseEvent`/`EventType`/`EventPublisher`，Kafka 实现 `KafkaEventPublisher`），现有模式是各服务定义 `XxxEventPublisher`（如 `order-service/.../OrderEventPublisher`）发 Kafka topic。RocketMQ `MQProducer`（common）无人 `start()`、`MQConsumer` 无调用方——不可作为发布通道
  - 改动范围：新增 `BlockchainReorgEvent`（extends `BaseEvent`）、扩展 `EventType` 枚举、新建 `BlockchainEventPublisher`（复用 Kafka）

- [ ] **通过 Kafka/RabbitMQ 等消息通道发布回滚通知，消息必须包含 `chain_id`、`common_ancestor`、`old_head`、`new_head`、`reorg_depth`**
  - 现状：Kafka 是事实事件总线（spring-kafka，各消费端 `@KafkaListener`）；RabbitMQ 无 Java 接入
  - 改动范围：复用 Kafka/event-sdk 发布（含 `chain_id/common_ancestor/old_head/new_head/reorg_depth`）；`application.yml` 确认 `spring.kafka.bootstrap-servers`（默认 `localhost:9092`，k8s 由环境变量覆盖）

- [ ] **提供内部管理 API：查询同步状态、当前 canonical head、最后确认区块、最近 reorg 和回滚范围**
  - 现状：无。现有管理入口可仿照 `wallet-service/.../interfaces/admin/WalletAdminController.java`
  - 改动范围：新建 sync 状态 controller（`/internal/chain/sync` 等）+ service；暴露状态机/游标/最近 reorg

- [ ] **提供受保护的重放 API/命令：按链、合约和区块范围重新扫描；支持 dry-run、断点续跑和幂等执行**
  - 现状：无任何重放能力
  - 改动范围：新建受保护 API/CLI + 断点游标 + 幂等执行器（配合 Phase 3 幂等键）

- [ ] **为下游消费者约定处理规范：收到回滚事件后按 `block_number > common_ancestor` 撤销，再消费新链事件**
  - 现状：目前无任何业务消费链上事件（data-pipeline 消费的是 order/inventory/user-behavior，非链上）
  - 改动范围：输出规范文档；未来接入的链上事件消费者统一按此实现

- [ ] **对无法回滚的外部副作用（通知、支付、发货、链下任务）增加确认门槛和补偿队列，不能在未确认区块上直接执行不可逆操作**
  - 现状：唯一不可逆外部副作用是 payment 外部转账：`CryptoWalletServiceImpl.transferCrypto`，`ExchangeOrderServiceImpl.processCryptoTransfer`（L211-226）**收到 HTTP 响应即置 `COMPLETED`**，无确认数等待
  - 改动范围：payment 增加确认门槛（确认数/finalized 后再置 `COMPLETED`/触发发货），补偿队列

## Phase 5: Graph Node 风格的确认与提交策略

- [ ] **将事件处理拆为“发现/暂存”和“确认/提交”两阶段，未达到确认条件的事件不得进入不可逆业务状态**
  - 现状：全部乐观单次写（`Solana*Service`），支付侧同步置 `COMPLETED`
  - 改动范围：写链路重构为 `PENDING → CONFIRMED → 投影提交`；未确认不写不可逆状态（listing Sold、activity 计数、payment COMPLETED）

- [ ] **支持配置 `reorg_threshold`，保留该范围内的区块和原始事件，确保发生 reorg 时能够回滚**
  - 现状：无配置项
  - 改动范围：新增配置 + `chain_block`/`chain_raw_event` 保留窗口策略（清理器禁用已确认区间的历史回滚能力）

- [ ] **对接 finalized/safe head 的链使用链自身最终性；不支持最终性的链使用确认数策略**
  - 现状：Solana 无 commitment 调用；EVM 无客户端
  - 改动范围：Solana `getBlock`/`getSignatureStatuses` 带 `commitment: "finalized"`；EVM client 以 finalized 标签为最终性来源

- [ ] **记录 `reorg_count`、`current_reorg_depth`、`max_reorg_depth`、`last_reorg_at` 等指标**
  - 现状：无指标记录。仓库未发现 Micrometer/Actuator exporter 配置
  - 改动范围：新增指标记录表或 Micrometer 指标 + 管理 API；确认 common 是否已引入 actuator 依赖（否则 pom 补）

- [ ] **对查询 API 明确返回数据所在区块号、区块 hash、确认状态和数据是否 finalized**
  - 现状：查询接口无 block 信息。涉及：`WalletQueryService`、`TraceabilityQueryService`、payment 订单查询、`SolanaTokenService.getToken`
  - 改动范围：查询 DTO 增加 `block_number/block_hash/is_confirmed/is_finalized`

- [ ] **防止查询层把 `REVERTED` 事件或非 canonical block 的投影返回给用户**
  - 现状：无 REVERTED/canonical 概念
  - 改动范围：查询层统一经过 canonical 过滤（配合 Phase 3 表状态）

## Phase 6: 测试与故障演练

- [ ] 使用本地链或测试网构造单区块 reorg，验证父哈希不匹配、共同祖先查找和自动恢复
  - 现状：无本地链端点配置。EVM 合约测试用 foundry（`contracts/evm/test/solidity`）与 hardhat；Solana 无本地 validator 配置
  - 改动范围：新增本地链编排（docker-compose 增加 anvil/anvil fork 或 `solana-test-validator`）+ RPC mock 支持（`SolanaRpcClient` 目前直接 `RestTemplate`，需允许注入 mock endpoint 或提炼接口）

- [ ] 构造多区块 reorg，验证多个区块、事件、实体和统计数据全部回滚
  - 现状：同上，无基础
  - 改动范围：同 6.1 + 业务投影回滚断言

- [ ] 验证 reorg 发生在事件暂存、业务提交、消息发送、服务重启和消费重试等不同阶段时的结果
  - 现状：无故障注入点
  - 改动范围：在各阶段边界加可测试性钩子

- [ ] 验证 RPC 节点切换、RPC 返回旧链、网络中断和 websocket 重连不会造成重复或漏事件
  - 现状：单 endpoint 无 failover；无 websocket
  - 改动范围：依赖 Phase 2.4 多 endpoint 成功后补测试

- [ ] 验证回滚消息重复投递、乱序投递和消费者重启时的幂等性
  - 现状：Kafka 消费端有但无链上消费者；可仿照 `data-pipeline/.../OrderEventConsumer` 的 `@KafkaListener` 写法做新消费端与测试
  - 改动范围：基于事件唯一键做幂等消费测试

- [ ] 增加端到端测试：新区块 → 事件入库 → reorg → 回滚 → 新链重放 → 查询结果一致
  - 现状：无（`e2e/` 目录无链上场景）
  - 改动范围：新增 e2e + 本地链编排成功的关键

- [ ] 增加数据库一致性检查脚本：检查业务数据是否引用了非 canonical block 或 `REVERTED` event
  - 现状：无
  - 改动范围：新建 SQL/巡检脚本（基于 `chain_block`/`chain_raw_event` 状态反查投影表 `source_event_id`/`block_hash`）

- [ ] 增加告警和 dashboard：同步延迟、当前 head、确认高度、回滚次数、回滚深度、重放耗时、死信数量
  - 现状：无 Prometheus/Grafana 配置；只有 k8s 基础资源
  - 改动范围：指标暴露 + 告警规则 + dashboard 配置（仓库可新增 `infra/` 下的监控编排）

- [ ] 在生产发布前执行一次可控 reorg 演练，并记录恢复时间和人工介入步骤
  - 现状：未做过
  - 改动范围：验收前执行，产出演练记录

## Phase 7: 交付验收标准

- [ ] 任意业务事件都能追溯到完整的区块 hash、交易 hash 和 log index
  - 差距：目前只有 `tx_signature`/`crypto_transaction_hash`，无 block/log 维度 → 依赖 Phase 1.3/1.6
- [ ] 区块父哈希不连续时，系统会停止提交并自动进入 reorg 处理，而不是继续写入
  - 差距：无门禁 → 依赖 Phase 2.2/2.3/2.6
- [ ] 被替换链上的数据不会出现在 canonical 查询结果中
  - 差距：无 canonical/REVERTED 概念 → 依赖 Phase 3/5.6
- [ ] 回滚后新链事件可以自动重放，且不会重复创建实体、重复累计统计或重复发送业务消息
  - 差距：无幂等 → 依赖 Phase 3.6/4.4
- [ ] 服务重启后能够从持久化 cursor 和 canonical block 表继续同步
  - 差距：无 cursor → 依赖 Phase 2.1
- [ ] 下游服务可以通过状态 API 或领域事件明确知道哪些区块已回滚
  - 差距：无 API/事件 → 依赖 Phase 4.1-4.3
- [ ] 超过自动回滚能力范围时系统会暂停并告警，而不是静默产生错误数据
  - 差距：无告警 → 依赖 Phase 2.8/5.4