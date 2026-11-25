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

- [x] 按「演示数据仅用于初始化／测试」的边界重建设备数据通路。

- [x] 建设协议无关的通用现场接入层（当前没有真实工厂，目标是后续最小修改）。
- [x] 实际接入 InfluxDB 时序库，不允许留空依赖。
- [x] 明确「数字影子 / 闭环孪生」边界，并实现受控写回（如需）。
- [x] 验证 zk + gateway + 数字孪生专用账号 token 的整条链路。
- [x] 补齐服务端事件 payload 并让前端无需回查。
