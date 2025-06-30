![阿卢说他遇到阿玮](https://tvax2.sinaimg.cn/crop.47.138.345.345.180/6b20647bly8fh6rmudt3cj20c80ha40r.jpg)

[nodejs文档](http://nodejs.cn/api/http.html)
[nest文档](https://docs.nestjs.com/support)

## Description

*本项目为元宇宙3D+区块链AI相关*

<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>

| 模块     | 方案                                 | 优势                 |
| ------ | ---------------------------------- | ------------------ |
| 前端     | Next.js + Tailwind CSS + shadcn/ui | SSR/SSG 全能选手，开发效率高 |
| 部署     | GitHub + Vercel                    | 自动构建、自动部署、全球 CDN   |
| 后端 API | Node.js/Java 自建服务器                      | 灵活、自由、成本低          |
| 数据库    | NoSQL → PlanetScale/PG                | 起步快，后期扩展轻松         |
| 用户系统   | Auth.js / Supabase                 | 简单开箱即用             |
| 邮件服务   | Resend                             | 接口简洁，免费额度友好        |
| App    | React Native                       | 代码复用，跨平台           |
| 高并发    | Redis+ES+K8s+RocketMQ                | 扩展选项         |


## Direct 目录说明

* block-chain 区块链
* server 后端、网关中心（协调、管理）
* evm-contract Ethereum及衍生链合约
* solana-contract Solana合约
* client 网站客户端（内容、浏览、支付）
* tools 工具库

## Installation 安装

```bash
$ yarn install
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
$ cd chain

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
