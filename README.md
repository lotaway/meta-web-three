## Description

_本项目为商城+AI风控+AI仓储+ERP+数字化工厂+（可选）区块链商品 相关_

## 目录结构

```
meta-web-three/
├── apps/              # 前端应用
├── server/            # 后端微服务
├── protos/            # Protocol Buffers 定义
├── contracts/         # 区块链智能合约
├── shared/            # 跨服务共享代码
├── infra/             # 基础设施配置
└── tools/             # 工具库
```

### apps/ 前端应用
- backstage-admin - 商城管理后台（后台管理系统，Vue 3 + Element Plus）
- client - 客户端 App（React Native / Expo，面向用户的移动端业务；包含支付、业务页面等）
- digital-twin - 数字孪生前端（AI 驱动的数字孪生与知识问答桌面端，Electron/React；并配套三维场景与交互）

### k8s/ 集群部署与新增服务

本仓库在 `k8s/` 目录提供了从 Docker Compose 迁移到 Kubernetes 的部署配置。

当你新增一个后端微服务（如 `server/<domain>/<service>/` 新增服务目录）时，通常还需要在 Kubernetes 侧补齐以下几类配置：
1. **业务服务资源**：在 `k8s/services/` 下新增该服务对应的 `*.yaml`（Deployment/Service/Ingress 等，参考已有 `product-service.yaml` 模板）。
2. **扩展领域服务汇总**：如果该服务属于“非基础内置集合”，需要把它加入 `k8s/services/extended-domain-services.yaml`（并最终通过 `k8s/deploy-all.yaml` 一键部署）。
3. **端口/访问入口**：确保该服务的端口在 `k8s/services/*.yaml` 中与 `server/<service>` 实际监听端口一致，并在需要对外暴露时配置 Ingress 路由（通常走 `k8s/api-gateway/` 或 `k8s/services/` 内的 ingress 资源）。
4. **前置的依赖与公共资源**：如果新服务依赖 MySQL/Redis/RabbitMQ 等基础组件，通常不需要新增 yaml；若你的服务需要额外 ConfigMap/Secret 或存储卷，再分别在 `k8s/configmaps/`、`k8s/secrets/`、`k8s/storage/` 中补齐。

更详细的映射规则与部署命令见：`k8s/README.md`。


### server/ 后端微服务

详见 [Server-README.md](./server/README.md)

### protos/ 消息定义
按领域拆分：mall/, supply-chain/, erp/, ai/, blockchain/, shared/

### contracts/ 区块链合约
- evm-contract - Ethereum 及 EVM 兼容链合约
- solana-contract - Solana 链合约

---

## Installation

```bash
$ yarn install
```

### AI 环境

需要以下 4 种的任意一种作为环境管理器使用：

- [anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive)
- [condaforce](https://conda-forge.org/miniforge)
- [uv](https://pypi.org/project/uv)
- pip

### Protobuf

需要 [protoc v32.0](https://github.com/protocolbuffers/protobuf/releases) 和 [CMake v4.1.0](https://cmake.org/download)。

macOS 安装：
```bash
brew install protobuf
brew install cmake
```

运行 `make` 生成多语言 protobuf 接口文件。

## Develop Guideline

* Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). 
* All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.
* After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/) or [Customer Client](apps/client/)