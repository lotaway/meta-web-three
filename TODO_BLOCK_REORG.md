
# 区块链事件同步与区块重组（Reorg）处理

> 目标：参考 Graph Node 的处理方式，使 `meta-web-three` 的区块链事件同步具备 canonical chain 校验、可回滚、可重放和下游一致性保障能力。所有链上事件写入业务库前，必须能够定位到 `block_number + block_hash + transaction_hash + log_index`，不能只依赖区块高度或交易哈希。

## Phase 1: 现状盘点与统一数据模型

- [ ] 梳理 `server/`、`protos/`、`contracts/`、消息队列和各业务服务中所有链上事件入口、订阅方式、落库表和派生统计表
- [ ] 梳理每条链的同步模式：RPC/WebSocket、轮询、第三方索引服务或 Firehose，并记录各自是否支持 `removed`/`revert`/`fork` 信号
- [ ] 统一定义链上事件元数据：`chain_id`、`contract_address`、`block_number`、`block_hash`、`parent_hash`、`transaction_hash`、`transaction_index`、`log_index`、`removed`、`observed_at`
- [ ] 在 `protos/blockchain/` 或共享模块中定义统一的 `BlockchainEvent`、`BlockRef`、`BlockReorg` 和同步进度消息
- [ ] 为事件增加唯一键：`chain_id + block_hash + transaction_hash + log_index`；禁止只用 `transaction_hash` 去重
- [ ] 为所有链上派生数据补充来源事件引用和区块元数据，至少包括 `source_event_id`、`block_number`、`block_hash`

## Phase 2: 区块同步与 canonical chain 校验

- [ ] 实现按区块同步的 `BlockCursor`，持久化最后处理区块的 `number`、`hash`、`parent_hash` 和同步状态
- [ ] 每处理新区块前校验：`new_block.parent_hash == cursor.block_hash`
- [ ] 发现父哈希不匹配时暂停业务事件提交，进入 reorg 检测流程，不得继续盲目消费新区块
- [ ] 实现从 RPC 获取区块头的能力：按高度和 hash 查询区块，并支持配置多个 RPC endpoint
- [ ] 实现共同祖先查找：从当前 cursor 和新链头逐级回溯，找到最后一个双方 hash 相同的 canonical block
- [ ] 增加同步状态机：`SYNCING`、`REORG_DETECTED`、`ROLLING_BACK`、`REPLAYING`、`HEALTHY`、`FAILED`
- [ ] 配置并记录确认策略：普通区块确认数、finalized/safe block（链支持时）、最大允许回滚深度
- [ ] 对超过最大回滚深度的情况告警并暂停自动提交，避免错误数据继续扩散

## Phase 3: 可回滚事件存储与业务数据回滚

- [ ] 建立 canonical block 表，保存区块号、区块 hash、父 hash、时间戳、状态和确认状态
- [ ] 建立原始链上事件表，保存完整事件元数据和原始 payload；原始事件与业务投影分离
- [ ] 为原始事件增加状态：`PENDING`、`CONFIRMED`、`REVERTED`、`PROCESSED`、`REPLAYED`
- [ ] 实现按共同祖先区块回滚：将祖先之后的 block/event 标记为 `REVERTED`，或在明确业务要求时删除
- [ ] 实现业务投影回滚：删除或撤销受影响事件产生的实体、余额、订单、统计、库存和聚合结果
- [ ] 对累加型统计禁止只做不可逆 `+/-` 猜测；优先支持按 canonical 事件重放重新计算
- [ ] 为动态数据、缓存、搜索索引、对象存储引用和异步任务建立回滚/补偿策略
- [ ] 确保回滚和重放具备幂等性，可重复执行而不会产生重复数据或重复扣减
- [ ] 使用事务或 outbox/inbox 机制保证“事件状态变化”和“业务投影变化”最终一致

## Phase 4: 事件发布、下游通知与重放

- [ ] 设计并实现 reorg 领域事件：`BlockReorgDetected`、`BlocksReverted`、`EventsReverted`、`ReplayStarted`、`ReplayCompleted`
- [ ] 通过 Kafka/RabbitMQ 等消息通道发布回滚通知，消息必须包含 `chain_id`、`common_ancestor`、`old_head`、`new_head`、`reorg_depth`
- [ ] 提供内部管理 API：查询同步状态、当前 canonical head、最后确认区块、最近 reorg 和回滚范围
- [ ] 提供受保护的重放 API/命令：按链、合约和区块范围重新扫描；支持 dry-run、断点续跑和幂等执行
- [ ] 为下游消费者约定处理规范：收到回滚事件后按 `block_number > common_ancestor` 撤销，再消费新链事件
- [ ] 对无法回滚的外部副作用（通知、支付、发货、链下任务）增加确认门槛和补偿队列，不能在未确认区块上直接执行不可逆操作

## Phase 5: Graph Node 风格的确认与提交策略

- [ ] 将事件处理拆为“发现/暂存”和“确认/提交”两阶段，未达到确认条件的事件不得进入不可逆业务状态
- [ ] 支持配置 `reorg_threshold`，保留该范围内的区块和原始事件，确保发生 reorg 时能够回滚
- [ ] 对接 finalized/safe head 的链使用链自身最终性；不支持最终性的链使用确认数策略
- [ ] 记录 `reorg_count`、`current_reorg_depth`、`max_reorg_depth`、`last_reorg_at` 等指标
- [ ] 对查询 API 明确返回数据所在区块号、区块 hash、确认状态和数据是否 finalized
- [ ] 防止查询层把 `REVERTED` 事件或非 canonical block 的投影返回给用户

## Phase 6: 测试与故障演练

- [ ] 使用本地链或测试网构造单区块 reorg，验证父哈希不匹配、共同祖先查找和自动恢复
- [ ] 构造多区块 reorg，验证多个区块、事件、实体和统计数据全部回滚
- [ ] 验证 reorg 发生在事件暂存、业务提交、消息发送、服务重启和消费重试等不同阶段时的结果
- [ ] 验证 RPC 节点切换、RPC 返回旧链、网络中断和 websocket 重连不会造成重复或漏事件
- [ ] 验证回滚消息重复投递、乱序投递和消费者重启时的幂等性
- [ ] 增加端到端测试：新区块 → 事件入库 → reorg → 回滚 → 新链重放 → 查询结果一致
- [ ] 增加数据库一致性检查脚本：检查业务数据是否引用了非 canonical block 或 `REVERTED` event
- [ ] 增加告警和 dashboard：同步延迟、当前 head、确认高度、回滚次数、回滚深度、重放耗时、死信数量
- [ ] 在生产发布前执行一次可控 reorg 演练，并记录恢复时间和人工介入步骤

## Phase 7: 交付验收标准

- [ ] 任意业务事件都能追溯到完整的区块 hash、交易 hash 和 log index
- [ ] 区块父哈希不连续时，系统会停止提交并自动进入 reorg 处理，而不是继续写入
- [ ] 被替换链上的数据不会出现在 canonical 查询结果中
- [ ] 回滚后新链事件可以自动重放，且不会重复创建实体、重复累计统计或重复发送业务消息
- [ ] 服务重启后能够从持久化 cursor 和 canonical block 表继续同步
- [ ] 下游服务可以通过状态 API 或领域事件明确知道哪些区块已回滚
- [ ] 超过自动回滚能力范围时系统会暂停并告警，而不是静默产生错误数据