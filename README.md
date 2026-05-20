## Description

_本项目为商城+AI风控+AI仓储+ERP+数字化工厂+（可选）区块链商品 相关_

## Direct Introduction 目录说明

- backstage-admin 商城管理后台
- client 客户端App
- server 商城后端
- protos protobuf RPC消息格式，提供给各个微服务使用
- risk-scorer 风险AI评分微服务
- evm-contract Ethereum 及衍生链合约
- solana-contract solana链合约
- solana-contract Solana链合约
- tools 工具库
- k8s k8s部署配置文件
- docker-* docker部署配置文件

## Installation 安装

```bash
$ yarn install
```

### AI环境

需要以下 4 种的任意一种作为环境管理器使用：

- [anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive)，第三方维护，包含非 py 包，数量几千，兼容性强
- [condaforce](https://conda-forge.org/miniforge)，社区维护，包含非 py 包，数量几万，但是兼容性差
- [uv](https://pypi.org/project/uv)，python 包管理器
- pip，python 环境自带

以 condaforce/miniforce 为例，下载后执行：

```bash
mamba create -n ai -c conda-forge
mamba activate ai
```

### protobuf

Need generate by [protoc v32.0](https://github.com/protocolbuffers/protobuf/releases) and [CMake v4.1.0](https://cmake.org/download) for `Rust` support, if need java grpc to build server, need install [protoc-gen-grpc-java v1.75.0](https://repo1.maven.org/maven2/io/grpc/protoc-gen-grpc-java), recommand uses dubbo to support grpc, that already inside java project dependency.
If using mac, can use `brew install protobuf` to install protoc, `brew install cmake` to install CMake.
After installed, run `make` to generate multiple language protobuf interface files.
