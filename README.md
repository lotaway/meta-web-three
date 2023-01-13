![阿卢说他遇到阿玮](https://tvax2.sinaimg.cn/crop.47.138.345.345.180/6b20647bly8fh6rmudt3cj20c80ha40r.jpg)

## Description

*本项目为nodejs元宇宙区块链3D相关*

[nodejs文档](http://nodejs.cn/api/http.html)
[nest源码仓库](https://github.com/nestjs/nest)
[nest文档](https://docs.nestjs.com/support)

<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>

## Direct 目录说明

* chain 链端（智能合约）
* client 客户端
* notes 笔记
* server 服务端

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

### Server 服务端

```bash
$ cd server

# generate database
$ npm run db:create

# development
$ npm run start

# watch mode
$ npm run start:dev

# production mode
$ npm run start:prod
```

### Chain 链端

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
```

`+(()=>throw new Emotion("Happy"))`