![阿卢说他遇到阿玮](https://tvax2.sinaimg.cn/crop.47.138.345.345.180/6b20647bly8fh6rmudt3cj20c80ha40r.jpg)

[nodejs文档](http://nodejs.cn/api/http.html)
[nest文档](https://docs.nestjs.com/support)

## Description

*本项目为元宇宙3D+区块链AI相关*

<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>

## Direct 目录说明

* block-chain 区块链
* block-chain-use 链端（合约）、网关中心（协调、管理）
* client 网站客户端（内容、浏览、支付）

## Installation 安装

```bash
$ npm install
```

## Running the app 同时运行服务端和客户端

```bash
npm run dev
```

### Client 客户端

```bash
$ cd client

# development
$ npm run dev
```

### Chain 链端 网关

```bash
$ cd chain

# compile contract
$ hardhat compile

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
