## Description

_本项目为元宇宙 3D+区块链 AI 相关_




| 模块     | 方案                               | 优势                                    |
| -------- | ---------------------------------- | --------------------------------------- |
| 前端     | Next.js + Tailwind CSS + shadcn/ui | SSR/SSG 全能选手，开发效率高            |
| 部署     | GitHub + Vercel                    | 自动构建、自动部署、全球 CDN            |
| 后端 API | Node.js 自建服务器                 | 灵活、自由、成本低，也可直接 Java 或 go |
| 数据库   | NoSQL → PlanetScale/PG             | 起步快，后期扩展轻松                    |
| 用户系统 | Auth.js / Supabase                 | 简单开箱即用                            |
| 邮件服务 | Resend                             | 接口简洁，免费额度友好                  |
| App      | React Native + Expo                | 代码复用，跨平台                        |
| Desktop  | Electron.js + Cpp                  | 跨平台，性能好，可快速迭代              |
| Game     | UE5                                | 3D引擎，画质精美                        |
| 高并发   | Redis+ES+K8s+RocketMQ              | 扩展选项                                |

## Direct 目录说明

- block-chain 区块链侧链
- server 后端，zk+dubbo+grpc+protobuf+spring cloud gateway+ micro services
- - common 公共模块
- - gateway 网关中心
- protos protobuf 协议文件
- order-match CEX订单撮合系统
- risk-scorer 风险评分AI服务
- evm-contract Ethereum 及衍生链合约
- solana-contract Solana 合约
- tools 工具库

## Installation 安装

```bash
$ yarn install
```

### ai 环境

需要以下 4 种的任意一种作为环境管理器使用：

- [anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive)，第三方维护，包含非 py 包，数量几千，兼容性强
- [condaforce](https://conda-forge.org/miniforge)，社区维护，包含非 py 包，数量几万，但是兼容性差
- [uv](https://pypi.org/project/uv)，python 包管理器
- pip，python 环境自带

以 anaconda 为例，下载后执行：

```bash
conda create -n ai -c conda-forge
conda activate ai
```

## Running the app 同时运行服务端和客户端

```bash
$ yarn dev
```

### Client 客户端

```bash
$ cd client

# development
$ yarn dev
```

### evm-contract 链端 网关

```bash
$ cd evm-contract

# compile contract
$ yarn compile

# contract test
$ hardhat test

# deploy contract
$ hardhat run scripts/deploy.js

# start a chain node
$ hardhat node

# deploy contract to local node for develop
$ hardhat run scripts/deploy.js --network localhost

# generate database
$ npm run db:generate

# development
$ npm run start

# watch mode
$ npm run start:dev

# production mode
$ npm run start:prod
```

`+(()=>throw new Emotion("Happy"))`

### protobuf

Need generate by [protoc v32.0](https://github.com/protocolbuffers/protobuf/releases) and [CMake v4.1.0](https://cmake.org/download) for `Rust` support, if need java grpc to build server, need install [protoc-gen-grpc-java v1.75.0](https://repo1.maven.org/maven2/io/grpc/protoc-gen-grpc-java), recommand uses dubbo to support grpc, that already inside java project dependency.
If using mac, can use `brew install protobuf` to install protoc, `brew install cmake` to install CMake.
After installed, run `make` to generate multiple language protobuf interface files.
