## Description

_本项目为商城+AI风控+AI仓储+ERP+数字化工厂+（可选）区块链商品 相关_

## 目录结构

```
meta-web-three/
├── apps/              # 前端应用
├── server/            # 后端微服务（详见 server/README.md）
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