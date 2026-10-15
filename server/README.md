# 后端 Server

此项目是zk+dubbo+grpc+protobuf+spring-cloud-gateway的micro-services

> **开发指引**：完整开发流程、代码规范、API 生成、测试等请参见 [DEVELOPMENT.md](../DEVELOPMENT.md)。

## Directory 目录结构

- - common 公共模块
- - gateway 网关中心
- - mall-domain/ 商城域
  - mall-catalog-service: 商品目录（product、mall-supplier、recommendation、review 合并）
  - mall-trade-service: 交易（order、payment、cart、promotion、after-sale 合并）
  - mall-user-service: 用户（user、user-action、risk-control 合并）
  - mall-social-service: 社交电商（live、group-buying、social-commerce 合并）
- - platform-domain/ 平台域
  - platform-service: 平台通用（message、media、commission、cs、data-analysis、tenant、developer-portal 合并）
- - factory-domain/ 工厂域
  - factory-service: 制造执行与数字孪生（mes、digital-twin、production 合并）
- - supply-chain-domain/ 供应链域
  - scm-inventory-service: 库存（inventory、warehouse、supplier、dom、inventory-alert 合并）
  - scm-logistics-service: 物流（logistics、procurement、rma 合并）
- - ai-domain/ AI 域
  - ai-service: AI 能力（forecasting、route-optimizer 合并）
- - erp-domain/ ERP 域
  - finance-service: 财务（finance、invoice、reporting、settlement 合并）
  - erp-management-service: 经营管理（crm、hrm、project 合并）
- - blockchain-domain/ 区块链域
  - blockchain-service: 链上能力（wallet、traceability 合并）
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

每个微服务的初始化表结构定义在 `src/main/resources/db/schema.sql`。已有数据库不要重复执行初始化脚本；本次 MES 补齐另提供手动增量脚本，部署方式见下方章节，未引入自动迁移框架。

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

## MES 持久化与安全拆单（factory-service）

安灯类型、等级、事件，设备类型、状态配置、转换规则及拆单规则已接入 PostgreSQL / MyBatis-Plus；这些是 MES 常见业务能力，不代表 ISA-95 强制规定对应仓储接口。生产注入使用持久化仓储，不以内存实现作为完成标准。

### 数据库部署

先备份目标数据库，并在与目标环境结构一致的测试库验证。于仓库根目录执行以下命令（连接信息使用本地 PostgreSQL 环境变量或受保护的凭据配置，不将密码提交到仓库）：

```bash
psql -v ON_ERROR_STOP=1 -f server/factory-domain/factory-service/src/main/resources/db/migration/V20261015__mes_configuration_and_safe_split.sql
```

脚本使用事务、`CREATE TABLE IF NOT EXISTS` 和增量索引，不删除或覆盖已有工单、设备表。它不是任意旧结构的自动修复器：已有表必须具备脚本所用字段；新增非空/唯一约束要求存量数据不存在空值或重复。若失败，应先核对结构、清理不合规数据，再重试，不要忽略错误。应用连接与迁移连接必须指向相同数据库和 schema。

### 后端入口与边界

以下为 factory-service 的直接接口路径；经过网关时按项目约定添加 `/factory-service` 前缀。

| 能力 | 入口 |
|------|------|
| 安灯配置与事件 | `/api/mes/andon/types`、`/levels`、`/events`；事件支持 acknowledge、handle、resolve、close、escalate 操作 |
| 设备配置 | `/api/mes/equipment-types`；类型下的 statuses、transitions、status-machine 与 active 配置 |
| 设备手动转换 | `POST /api/mes/equipment/{id}/transition`，请求体为 `{"targetStatus":"目标状态编码"}` |
| 拆单规则 | `POST/GET /api/mes/work-order-splits/rules` |
| 拆单与子单查询 | `POST /api/mes/work-order-splits/{parentId}`、`GET /api/mes/work-order-splits/{parentId}/children` |

拆单仅支持 `BY_QUANTITY` 和 `MANUAL`，仅允许已保存的草稿工单；子单数量为正、总量守恒，保留父单原始数量和子单关联，父单转为 `CANCELLED` 防止重复执行。事务保证失败不留下部分子单，父单锁与唯一索引防止并发重复拆分。`BY_BOM`、`BY_PROCESS` 和旧合并逻辑明确拒绝执行，真实 BOM 展开、工序任务分解与父子完工汇总仍见根目录 TODO。

设备转换首先校验类型有效性；已绑定通用状态机时以该状态机为规则来源，未绑定时使用设备类型专属状态配置和转换规则，两者不叠加。未实现执行器的条件表达式、事件动作及自动/超时转换不能被手动入口绕过。本次未新增前端页面，也未完成目标环境部署和网关端到端验收。

### 验证

```bash
mvn -f server/factory-domain/factory-service/pom.xml test -DskipTests=false
```

真实 PostgreSQL 测试需设置 `MES_TEST_URL`（不含查询参数的 JDBC URL）、`MES_TEST_USER`、`MES_TEST_PASSWORD`，账号需有创建 schema 权限。每次运行创建并保留独立 `mes_verify_*` schema，不使用业务表；未设置 URL 时数据库测试会跳过。已验证 7 项拆单测试及 8 项 PostgreSQL/接口测试，包括迁移重复应用、持久化重载、安灯生命周期、设备规则、数量守恒、拒绝路径、事务回滚与并发防重复。这不等同于完整 Spring Boot 服务启动或生产验收。

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

