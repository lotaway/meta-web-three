# Solana 商城模板集成方案 (状态: P1-P3 全部完成 ✅)

## 概述

将现有 EVM 合约的商城功能移植到 Solana 区块链。P1-P3 所有子任务已完成，包含合约、后端服务、前端页面。P0 (合约编译部署) 待做。

## 实现状态

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| Token/NFT/SFT 创建 | ✅ 完成 | `create_token`, `create_sft`, `mint_to`, `burn_tokens` |
| 商城销售 (List/Buy/Delist) | ✅ 完成 | `list_good(price, listed_amount)`, `buy_good`, `delist_good` |
| 活动 (创建/参与/领奖) | ✅ 完成 | Merkle proof 验证奖励分配 |
| 佣金 (上下线关系) | ✅ 完成 | `set_upline` 上级绑定, 防自引用 |

## Bug 修复 (2025-07-24)

| # | 文件 | 问题 | 修复 |
|--|------|------|------|
| 1 | `context.rs:189` | `Activity` 同时有 `#[derive(Accounts)]` 和 `#[account]` | 移除 `#[derive(Accounts)]` |
| 2 | `context.rs:322` | `DelistGood` 缺少 `#[derive(Accounts)]` | 添加派生属性 |
| 3 | `context.rs` | `CreateToken` 结构体缺失 (lib.rs 引用但未定义) | 添加 `CreateToken` 账户结构 |
| 4 | `context.rs:76` | `Listing` 缺少 `listed_amount` 字段 | 添加字段 + 更新 `LEN` |
| 5 | `lib.rs:256` | `list_good` 将 `price` 转入托管 (应转 `listed_amount`) | 改为 `listed_amount` 参数 |
| 6 | `lib.rs:282` | `buy_good` 从托管转出 `price` (应转 `listed_amount`) | 改为 `listing.listed_amount` + 设置 `status=1` |
| 7 | `lib.rs:319` | `delist_good` 从托管转出 `price` | 改为 `listing.listed_amount` + 设置 `status=2` |

## 合约指令签名

| 指令 | 参数 | 说明 |
|------|------|------|
| `initialize` | — | 初始化程序 |
| `create_token_and_nft` | `(name, symbol, uri)` | 创建 NFT (decimals=0) |
| `create_token` | `(name, symbol, uri, supply)` | 创建 FT (decimals=9) |
| `create_sft` | `(name, symbol, uri, supply)` | 创建 SFT (decimals=0, is_sft=true) |
| `mint_to` | `(amount)` | 增发代币 |
| `burn_tokens` | `(amount)` | 销毁代币 |
| `deposit` | `(amount)` | 存入 Program Token Account |
| `withdraw` | `(amount)` | 从 Program Token Account 提现 |
| `list_good` | `(price, listed_amount)` | 上架商品 (NFT/SFT 转入托管) |
| `buy_good` | — | 购买 (支付 → 卖家, 托管 → 买家, status→1) |
| `delist_good` | — | 下架 (托管 → 卖家, status→2) |
| `create_activity` | `(start_time, end_time, entry_fee, reward_pcts[3])` | 创建活动 |
| `participate_activity` | — | 参与活动 (支付 entry fee) |
| `set_merkle_root` | `(root[32])` | 设置 Merkle root |
| `claim_reward` | `(rank, proof[])` | 领奖 (Merkle 验证) |
| `set_upline` | — | 设置上级 (防自引用) |
| `distribute_commission` | `(sale_amount)` | 佣金自动分配 (10% 给上级, 需 seller 签名 + upline ATA 为剩余账户) |
| `create_coupon` | `(discount_amount, max_uses, merkle_root, expiry)` | 创建优惠券 (Merkle tree) |
| `redeem_coupon` | `(proof)` | 兑换优惠券 (Merkle 验证 + 从池转账) |

## 文件变更清单 (最终)

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `server/.../wallet/infrastructure/solana/SolanaRpcClient.java` | Solana RPC 客户端 |
| `server/.../wallet/application/dto/CreateTokenRequest.java` | 创建代币 DTO |
| `server/.../wallet/application/dto/MintTokenRequest.java` | 增发 DTO |
| `server/.../wallet/application/dto/SolanaTokenDTO.java` | 代币信息 DTO |
| `server/.../wallet/application/dto/ListingRequest.java` | 上架请求 DTO |
| `server/.../wallet/application/dto/ListingDTO.java` | 上架信息 DTO |
| `server/.../wallet/application/dto/BuyRequest.java` | 购买请求 DTO |
| `server/.../wallet/application/dto/CreateActivityRequest.java` | 创建活动 DTO |
| `server/.../wallet/application/dto/ActivityDTO.java` | 活动信息 DTO |
| `server/.../wallet/application/dto/CommissionDTO.java` | 佣金信息 DTO |
| `server/.../wallet/application/service/SolanaTokenService.java` | 代币服务 |
| `server/.../wallet/application/service/SolanaMarketplaceService.java` | 商城服务 |
| `server/.../wallet/application/service/SolanaActivityService.java` | 活动服务 |
| `server/.../wallet/application/service/SolanaCommissionService.java` | 佣金服务 |
| `server/.../wallet/interfaces/controller/SolanaTokenController.java` | Token API |
| `server/.../wallet/interfaces/controller/SolanaMarketplaceController.java` | 商城 API |
| `server/.../wallet/interfaces/controller/SolanaActivityController.java` | 活动 API |
| `server/.../wallet/interfaces/controller/SolanaCommissionController.java` | 佣金 API |
| `apps/backstage-admin/src/apis/solana.ts` | 前端 API |
| `apps/backstage-admin/src/views/solana/token/index.vue` | 代币页面 |
| `apps/backstage-admin/src/views/solana/marketplace/index.vue` | 商城页面 |
| `apps/backstage-admin/src/views/solana/activity/index.vue` | 活动页面 |
| `apps/backstage-admin/src/views/solana/commission/index.vue` | 佣金页面 |
| `contracts/solana/tests/solana-contract.ts` | 合约测试 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `contracts/solana/programs/solana-contract/src/lib.rs` | 修复 withdraw bug + 添加所有新指令 + 修复 listed_amount bug |
| `contracts/solana/programs/solana-contract/src/context.rs` | 添加所有账户结构 + 修复 Activity/DelistGood 属性 + 添加 CreateToken |
| `contracts/solana/programs/solana-contract/src/seeds.rs` | 添加 TOKEN_CONFIG, MINT_AUTHORITY, LISTING, LISTING_ESCROW, ACTIVITY, COMMISSION |
| `contracts/solana/Cargo.toml` | 添加 mpl-token-metadata 依赖 |
| `apps/backstage-admin/src/router/index.ts` | 添加 Solana 路由 |
| `apps/backstage-admin/src/locales/en-US.ts` | 补齐 i18n |
| `apps/backstage-admin/src/locales/zh-CN.ts` | 补齐 i18n |
| `SOLANA_PLAN.md` | 本文档 (状态追踪) |
| `server/.../domain/entity/SolanaKeypair.java` | KMS 加密密钥对实体 |
| `server/.../repository/SolanaKeypairMapper.java` | KMS Mapper |
| `server/.../solana/SolanaWalletManager.java` | AES-256-GCM 钱包管理器 |
| `server/.../controller/SolanaWalletController.java` | 钱包管理 API (生成/导入/列表) |
| `contracts/solana/.../context.rs` | 新增 CommissionGraph, SetUpline, DistributeCommission 结构 |
| `contracts/solana/.../lib.rs` | 新增 distribute_commission 指令 + UplineNotSet 错误 |
| `server/.../solana/SolanaContractClient.java` | 新增 distributeCommission, createCoupon, redeemCoupon, deriveCouponAddress, deriveCouponPoolAddress 方法 |
| `contracts/solana/.../seeds.rs` | 新增 COUPON 种子 |
| `server/.../dto/CouponDTO.java` | 优惠券 DTO |
| `server/.../service/SolanaCouponService.java` | 优惠券服务 |
| `server/.../controller/SolanaCouponController.java` | 优惠券 API |

## 待办项 (按优先级)

### P0: 合约编译和部署
- [ ] 安装 Anchor 框架 / Solana CLI
- [ ] 运行 `anchor build` 编译合约
- [ ] 运行 `anchor test` 执行测试

### P1: 真实交易集成 (后端)
- [x] 实现 `SolanaContractClient.java` — 交易构建和签名
- [x] 替换后端服务中的模拟/PENDING 响应
- [x] 添加密钥管理 (AES-256-GCM KMS, 本地数据库存储加密私钥)

### P2: 链下数据库表
- [x] `tb_solana_listing` — 商品上架元数据
- [x] `tb_solana_activity` — 活动数据
- [x] `tb_solana_commission_relation` — 上下线关系

### P3: 功能增强
- [x] 佣金自动分配 (`distribute_commission` 指令) — 合约 + 后端 API
- [x] 优惠券功能 (MerkleProof 折扣) — Coupon 账户 + create_coupon / redeem_coupon 指令
- [x] 活动详情页 (`views/solana/activity/detail.vue`)
- [x] 我的上架管理 (`views/solana/marketplace/my-listings.vue`)

- [x] 给 backstage-admin 所有关于区块链的功能都需添加一个配置控制，通过.env或者其他开发运维能修改的方式决定是否显示/隐藏，并且默认值是隐藏