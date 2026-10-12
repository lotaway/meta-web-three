# 后端 Server

此项目是zk+dubbo+grpc+protobuf+spring-cloud-gateway的micro-services

> **开发指引**：完整开发流程、代码规范、API 生成、测试等请参见 [DEVELOPMENT.md](../DEVELOPMENT.md)。

## Directory 目录结构

- - common 公共模块
- - gateway 网关中心
- - mall-domain/ 商城域（mall-catalog-service、mall-trade-service、mall-user-service、mall-social-service）
- - platform-domain/ 平台域（platform-service）
- - factory-domain/ 工厂域（factory-service）
- - supply-chain-domain/ 供应链域（scm-inventory-service、scm-logistics-service）
- - ai-domain/ AI 域（ai-service）
- - erp-domain/ ERP 域（finance-service、erp-management-service）
- - blockchain-domain/ 区块链域（blockchain-service）
- - data-pipeline/ 数据管道（独立模块，不属于任何域）

本地多服务启动：`./run-server.sh`（服务列表见 `scripts/server-services-registry.sh`）。

Docker / Compose 构建（上下文为**仓库根目录**）：

```bash
docker build -f server/Dockerfile --target mall-catalog-service .
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d mall-catalog-service
```

端口分配（合后 14 个 Java 服务）：

| 端口 | 服务 | 域 |
|------|------|-----|
| 10081 | gateway | gateway |
| 10082 | mall-catalog-service | mall |
| 10083 | mall-user-service | mall |
| 10084 | mall-trade-service | mall |
| 10085 | platform-service | platform |
| 10095 | mall-social-service | mall |
| 10101 | factory-service | factory |
| 10103 | ai-service | ai |
| 10105 | scm-inventory-service | supply-chain |
| 10107 | scm-logistics-service | supply-chain |
| 10110 | finance-service | erp |
| 10114 | blockchain-service | blockchain |
| 10122 | data-pipeline | 数据管道（独立模块） |
| 10125 | erp-management-service | erp |

> 完整的 46 → 15 合并映射见 [docs/microservice-consolidation.md](../docs/microservice-consolidation.md)。

K8s 扩展服务：`kubectl apply -f k8s/services/extended-domain-services.yaml`

## 微服务功能说明

### server - 父项目

主要负责引入所有基础依赖库

### common - 公共模块
**职责**: 跨服务共享的基础能力，也负责增强依赖库+公共配置
- 统一错误处理
- 通用工具类
- 常量定义
- 公共配置
- 分布式 ID 生成器

### gateway - API 网关
**职责**: 统一 API 入口，负责请求路由、负载均衡、限流、鉴权、日志
- 请求路由与转发
- JWT 令牌验证
- 接口限流与熔断
- 统一响应格式
- 请求日志记录

### mall-catalog-service - 商品目录服务（mall 域）
**职责**: 商品与类目核心（合并原 product、mall-supplier、recommendation、review 服务）
- 商品发布与编辑、商品类目管理、SKU/SPU 管理
- 商品搜索与过滤、商品评价与问答
- 供应商管理
- AI 辅助购物（文本纠错、智能匹配、以图搜图、一站式搜索，见下方专节）

### mall-trade-service - 交易服务（mall 域）
**职责**: 交易核心（合并原 order、payment、cart、promotion、after-sale 服务）
- 订单创建与状态管理（待支付、已支付、已发货、已完成、已取消）
- 订单取消与退款、订单 History、逆向流程（退货、退款）
- 购物车管理（加购、批量操作、数量修改、勾选结算、登录后合并）
- 支付通道集成（微信 / 支付宝 / Stripe）、支付回调与退款
- 营销活动（优惠券、满减、限时折扣、拼团、秒杀）与营销规则计算
- **注意**: 订单不直接操作库存，通过事件调用 scm-inventory-service

### mall-user-service - 用户服务（mall 域）
**职责**: 用户体系（合并原 user、user-action、risk-control 服务）
- 注册与登录（手机号、邮箱、第三方 OAuth）、用户信息 CRUD
- 用户地址管理、会员等级与积分、用户标签与画像
- 用户行为采集与分析（浏览记录、收藏、足迹、搜索历史）
- 风险控制

### mall-social-service - 社交电商服务（mall 域）
**职责**: 社交电商互动（合并原 live、group-buying、social-commerce 服务）
- 直播、拼团、社交电商等互动能力（REST 控制器 + admin 接口，无 GraphQL 子图）

### data-pipeline - 数据管道服务
**职责**: ETL 数据处理与 OLAP 分析，负责消费事件、写入 ClickHouse
- 消费 Kafka/RocketMQ 事件（订单、库存、用户行为）
- 数据转换与清洗（维度字段补充）
- 写入 ClickHouse 分析表
- OLAP 多维查询接口
- 数据血缘追踪（lineage）
- WebSocket 实时看板推送
- **注意**: 该模块为独立服务，不继承 common 模块的默认配置；依赖 ClickHouse 作为分析数据库

### platform-service - 平台服务（platform 域）
**职责**: 平台通用能力（合并原 message、media、commission、cs、data-analysis、tenant、developer-portal 服务）
- 消息通知（站内信、短信、邮件、模板与渠道配置）
- 文件与媒体管理（图片处理、视频转码、OSS 对象存储、CDN）
- 分销与佣金（计算规则、结算、分销关系、提现）
- 客服会话（含 WebSocket）、数据分析、租户管理、开发者门户

### scm-inventory-service - 库存服务（supply-chain 域）
**职责**: 库存与仓储（合并原 inventory、warehouse、supplier、dom、inventory-alert 服务）

### scm-logistics-service - 物流服务（supply-chain 域）
**职责**: 物流与采购（合并原 logistics、procurement、rma 服务）

### finance-service - 财务服务（erp 域）
**职责**: 财务与结算（合并原 finance、invoice、reporting、settlement 服务）

### erp-management-service - 经营管理服务（erp 域）
**职责**: 企业经营管理（合并原 crm、hrm、project 服务）
- 客户关系管理（线索池、商机、客户联系人、工单、营销活动、看板）
- 人力资源（HRM）
- 项目管理（控制器 base path 为 `/project-service/*`）

### factory-service - 工厂服务（factory 域）
**职责**: 制造执行与数字孪生（合并原 mes、digital-twin、production 服务）

### ai-service - AI 服务（ai 域）
**职责**: AI 能力服务（合并原 forecasting、route-optimizer 服务）

### blockchain-service - 区块链服务（blockchain 域）
**职责**: 链上能力（合并原 wallet、traceability 服务）

## AI 辅助购物（mall-catalog-service）

AI 辅助购物能力已随 `recommendation-service` 并入 `mall-catalog-service`（端口 10082）：**文本纠错**、**智能匹配**、**以图搜图**、一站式搜索。

- **路由**：所有 REST 请求以 gateway 为入口、配合 ZK 服务发现（discovery locator）转发，URL 需带服务名前缀 `/mall-catalog-service/...`；C 端和后台的前端代码已按此约定接入（后台 `http` 工具通过 `SERVICE_PREFIX_MAP` 自动添加前缀）。
- **AI Provider**：OpenAI 兼容协议（`/v1/embeddings`、`/v1/images/embeddings`、`/v1/chat/completions`），`base-url/api-key/model` 在 `application.yml` 的 `ai-shopping.*` 下配置，可被 `ai_shopping_config` 表（DB）覆盖，后台页面可维护。
- **向量库**：Milvus（REST API）；`vector.store=memory` 时用内存向量库兜底（默认）。
- **Milvus 网络**：`docker-compose.ai.yml` 已纳入主 compose `include`，etcd/minio/milvus 统一挂载到 `meta-web-three` 网络，与后端服务同网互访（`milvus:19530`）。

## 添加新服务清单

添加一个新微服务时，需按以下清单逐项完成，否则会导致构建或运行失败：

1. `protos/` 下新建 `.proto` 文件定义服务接口和消息结构
2. 在项目根目录执行 `make gen`，借助 `Makefile` 生成 Java 胶水代码到 `common` 模块
3. 在 `server/<domain>/<service>/` 下新建 Spring Boot 项目，实现 `common` 中的接口；**`application.yml` 必须参照 `server/_templates/application-service.yml` 模板编写**
4. 调用方通过以下方式引用该服务：
   - Java 服务：调用方使用 `@DubboReference(check = false, lazy = true)` 注入服务并调用服务接口方法
   - 非 Java 服务：通过 ZK + gRPC 方式，结合生成的 proto 代码进行调用
5. 在 `scripts/server-services-registry.sh` 中按 `服务名|模块路径(相对server/)|HTTP端口` 格式追加一行注册服务
6. 若新服务暴露 GraphQL 端点（`/graphql`），在 `gateway/.../graphql/FederationRouter.java` 中的以下四处注册：
   - `SUBGRAPH_URLS`：添加 `服务名 → http://<service-name>/graphql` 映射
   - `ROOT_FIELD_OWNER`：添加该服务暴露的所有根查询/变更字段
   - `ENTITY_TYPES`：添加该服务定义的 Federation entity 类型
   - `buildCombinedSdl()`：添加该服务的 GraphQL 类型定义
7. 在 `server/Dockerfile` 中新增 `FROM eclipse-temurin:17-jre-jammy AS <service-name>` 构建阶段
8. 在 `docker-compose.server.yml` 中新增 service 定义（参考已有服务模板 `<<: *java-service`），含 `hostname`、`ports`、`volumes` 等
9. 对照下方端口表分配未占用的端口（10101+ 为新增领域服务）
10. 在 `allow-ports-firework.sh` 的 `PORTS` 数组中添加该服务的端口号
11. **K8s 配置最简补齐**：为该服务添加 Kubernetes 部署/访问 yaml（放在 `k8s/services/`），并在需要的话加入 `k8s/services/extended-domain-services.yaml` 以纳入一键部署；最终以 `k8s/deploy-all.yaml` 为入口完成落地。

说明：
- `run-server.sh` 会自动读取 `scripts/server-services-registry.sh` 中的服务列表，无需手动修改
- 服务的 `spring.application.name` 必须与 `SUBGRAPH_URLS` 中的服务名一致，才能被 `@LoadBalanced RestTemplate` 通过 Zookeeper 服务发现正确路由
- 服务暴露 GraphQL Federation 端点时，需在 `gateway` 的 `pom.xml` 中添加对应的 `common` 模块 RPC client（如已有则跳过），然后在 `gateway/client/` 下添加相应的 `@DubboReference` Client 类

## 微服务必填配置清单

以下配置项在各微服务的 `application.yml`（或 profile 文件）中**无有效默认值**（占位符 `your_*` / 空值），启动服务前必须替换为真实值，否则对应功能将运行失败。

| 微服务 | 配置项 | 当前值 | 说明 |
|--------|--------|--------|------|
| mall-trade-service | `payment.fiat.alipay.app-id` | `your_alipay_app_id` |  支付宝 AppID，占位符 |
| mall-trade-service | `payment.fiat.alipay.private-key` | `your_alipay_private_key` | 支付宝应用私钥（RSA2），占位符 |
| mall-trade-service | `payment.fiat.alipay.public-key` | `your_alipay_public_key` | 支付宝公钥，占位符 |
| mall-trade-service | `payment.fiat.wechat.app-id` | `your_wechat_app_id` | 微信 AppID，占位符 |
| mall-trade-service | `payment.fiat.wechat.mch-id` | `your_wechat_mch_id` | 微信商户号，占位符 |
| mall-trade-service | `payment.fiat.wechat.api-key` | `your_wechat_api_key` | 微信 APIv2 密钥，占位符 |
| mall-trade-service | `payment.fiat.wechat.cert-path` | `/path/to/wechat/cert.p12` | 微信商户证书路径，占位符 |
| mall-trade-service | `payment.crypto.wallet.api.key` | `your_api_key` | 区块链钱包 API Key，占位符 |
| mall-trade-service | `payment.crypto.wallet.hot-wallet.*` | `your_*_hot_wallet_address` | 热钱包地址（BTC/ETH/USDT），占位符 |
| mall-trade-service | `payment.crypto.wallet.cold-wallet.*` | `your_*_cold_wallet_address` | 冷钱包地址（BTC/ETH/USDT），占位符 |
| mall-trade-service | `payment.reconciliation.payment-platform-api-url` | 空字符串 | 对账平台 API 地址为空；若启用对账功能需配置 |
| mall-trade-service | `payment.reconciliation.message-service-url` | 空字符串 | 消息服务地址为空；若启用对账通知需配置 |
| mall-trade-service | `payment.reconciliation.dingtalk-webhook-url` | 空字符串 | 钉钉告警 Webhook 为空；若启用对账告警需配置 |
| blockchain-service | `blockchain.evm.rpc-url` | `https://mainnet.infura.io/v3/YOUR_PROJECT_ID` | Infura Project ID `YOUR_PROJECT_ID` 为占位符 |
| mall-user-service | `x.apikey` | `your-api-key` | `mall-user-service/src/main/resources/application.yml` 中的占位符 |
| mall-user-service | `x.secret` | `your-secret-key` | 同上 |
| mall-catalog-service | `ai-shopping.embedding.base-url` / `api-key` / `model` | 空字符串 | AI 智能匹配文本向量化端点；未配置则智能匹配不可用 |
| mall-catalog-service | `ai-shopping.image-embedding.base-url` / `api-key` / `model` | 空字符串 | AI 以图搜图图像向量化端点；未配置则以图搜图不可用 |
| mall-catalog-service | `ai-shopping.llm.base-url` / `api-key` / `model` | 空字符串 | 文本纠错 LLM 端点；未配置时自动降级为本地词典纠错 |
| data-pipeline | `clickhouse.url` | `jdbc:clickhouse://localhost:8123/meta_web_analytics` | ClickHouse JDBC 连接 URL；Docker 内需改为 `jdbc:clickhouse://clickhouse:8123/meta_web_analytics` |

### 其他注意事项

- 所有微服务统一使用 **PostgreSQL**作为关系型数据库，禁止使用 MySQL。common 模块已配置 PostgreSQL 驱动和默认连接，各服务无需重复声明 datasource（除非需要多数据源）。
- **data-pipeline**：已依赖 common 模块，继承公共配置（datasource、Redis、Dubbo 等）；Kafka、RocketMQ 消费者组、ClickHouse 连接信息在 `application.yml` 默认 profile 中配置。
- **ClickHouse**：作为 data-pipeline 的分析数据库，已纳入 `docker-compose.env.yml`；启动前 `.env` 中 `CLICKHOUSE_HOST` / `CLICKHOUSE_PORT` / `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` 可留空使用默认值。

## AWS S3 Configuration

To use AWS S3, you need to configure the following properties in `.aws/credentials` file in the root directory of your project:

```bash
[default]
aws_access_key_id=s3_access_key
aws_secret_access_key=s3_secret_key
```

## Quartz Schedule Config

[Quartz SQL Scripts and Example](https://www.quartz-scheduler.org/downloads)

## Database Schema

每个微服务的数据库表结构定义在 `src/main/resources/db/schema.sql`（统一只使用初始化 schema，不使用版本迁移脚本）。

### 服务目录结构

```
<domain>/<service>/
├── src/main/resources/db/
│     └── schema.sql      -- 表结构定义（初始化脚本）
```

### ID 生成策略

多微服务环境下，统一使用代码生成 ID（雪花算法），避免数据库自增冲突：

- **IdType.ASSIGN_ID**: 使用 MyBatis-Plus 雪花算法生成 ID
- **IdType.INPUT**: 手动设置 ID（如订单号等业务主键）

禁止使用 `IdType.AUTO`（数据库自增）。

## 支付模块配置

### 支付方式

| 支付方式 | SDK | 配置文件 |
|---------|-----|---------|
| 微信支付 | wechatpay-sdk-java | `mall-trade-service/src/main/resources/application.yml` |
| 支付宝 | alipay-sdk-java | `mall-trade-service/src/main/resources/application.yml` |
| Stripe | stripe-java | `mall-trade-service/src/main/resources/application.yml` |

### 配置步骤

#### 1. 修改 application.yml

在 `mall-trade-service/src/main/resources/application.yml` 中配置：

```yaml
payment:
  fiat:
    # 支付宝配置
    alipay:
      app-id: your_alipay_app_id
      private-key: your_alipay_private_key
      public-key: your_alipay_public_key
      gateway-url: https://openapi.alipay.com/gateway.do
    
    # 微信支付配置
    wechat:
      app-id: your_wechat_app_id
      mch-id: your_wechat_mch_id
      api-key: your_wechat_api_key
      cert-serial-number: your_cert_serial_number
      private-key: your_wechat_private_key
    
    # Stripe 配置
    stripe:
      secret-key: sk_test_your_stripe_secret_key
```

#### 2. 微信支付配置说明

- **app-id**: 微信公众平台 AppID
- **mch-id**: 商户号
- **api-key**: APIv2 密钥（用于签名）
- **cert-serial-number**: 证书序列号（用于 v3 API）
- **private-key**: 商户私钥（用于 v3 API 签名）

#### 3. 支付宝配置说明

- **app-id**: 支付宝应用 AppID
- **private-key**: 应用私钥（RSA2）
- **public-key**: 支付宝公钥

#### 4. Stripe 配置说明

- **secret-key**: Stripe Secret Key（测试/生产）

### 接口列表

| 接口 | 路径 | 说明 |
|------|------|------|
| 微信支付参数 | `POST /api/pay/wechat/params` | 返回调起微信支付所需参数 |
| 支付宝参数 | `POST /api/pay/alipay/params` | 返回支付宝 orderString |
| Stripe参数 | `POST /api/pay/stripe/params` | 返回 Stripe clientSecret |
| 支付验证 | `POST /api/pay/verify` | 验证支付结果 |

### 回调配置

需要在支付平台配置回调地址：

| 支付方式 | 回调地址 |
|---------|---------|
| 微信支付 | `https://your-domain.com/api/pay/wechat/callback` |
| 支付宝 | `https://your-domain.com/api/pay/alipay/callback` |

