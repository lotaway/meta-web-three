# 后端 Server

此项目是zk+dubbo+grpc+protobuf+spring-cloud-gateway的micro-services

## Directory 目录结构

- - common 公共模块
- - gateway 网关中心
- - mall-domain/ 商城域（product、order、user、payment、cart、promotion-service）
- - platform-domain/ 平台域（message、media、commission、user-action、cs-service）
- - factory-domain/ 工厂域
- - supply-chain-domain/ 供应链域
- - ai-domain/ AI 域
- - erp-domain/ ERP 域
- - blockchain-domain/ 区块链域

本地多服务启动：`./run-server.sh`（服务列表见 `scripts/server-services-registry.sh`）。

Docker / Compose 构建（上下文为**仓库根目录**）：

```bash
docker build -f server/Dockerfile --target product-service .
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d product-service
```

端口分配（10101+ 为新增领域服务）：

| 端口 | 服务 | 域 |
|------|------|-----|
| 10081-10092 | gateway、商城、平台核心服务 | gateway / mall / platform |
| 10101-10104 | mes、digital-twin、forecasting、recommendation | factory / ai |
| 10105-10109 | inventory、warehouse、logistics、procurement、supplier | supply-chain |
| 10110-10113 | finance、invoice、reporting、settlement | erp |
| 10114 | wallet | blockchain |

K8s 扩展服务：`kubectl apply -f k8s/services/extended-domain-services.yaml`

## 微服务功能说明

### gateway - API 网关
**职责**: 统一 API 入口，负责请求路由、负载均衡、限流、鉴权、日志
- 请求路由与转发
- JWT 令牌验证
- 接口限流与熔断
- 统一响应格式
- 请求日志记录

### user-service - 用户服务
**职责**: 用户体系管理，负责用户注册、登录、信息管理
- 用户注册与登录（手机号、邮箱、第三方 OAuth）
- 用户信息 CRUD
- 用户地址管理
- 会员等级与积分
- 用户标签与画像

### product-service - 商品服务
**职责**: 商品核心业务，负责商品发布、类目、SKU 管理
- 商品发布与编辑
- 商品类目管理
- SKU/SPU 管理
- 商品搜索与过滤
- 商品库存查询（仅查询，不管理）
- 商品评价与问答

### order-service - 订单服务
**职责**: 交易核心，负责订单创建、状态流转、取消退款
- 订单创建与支付
- 订单状态管理（待支付、已支付、已发货、已完成、已取消）
- 订单取消与退款
- 订单 History 记录
- 订单逆向流程（退货、退款）
- **注意**: 订单服务不直接操作库存，通过事件调用 inventory-service

### cart-service - 购物车服务
**职责**: 购物车管理，负责加购、批量操作
- 添加/删除商品
- 数量修改
- 勾选结算
- 购物车合并（登录后）

### payment-service - 支付服务
**职责**: 支付通道集成，负责对接第三方支付
- 微信支付集成
- 支付宝集成
- Stripe 支付集成
- 支付结果回调处理
- 支付状态查询
- 退款处理

### promotion-service - 营销服务
**职责**: 营销活动管理，负责优惠券、满减、折扣等活动
- 优惠券创建与发放
- 满减活动配置
- 限时折扣
- 拼团活动
- 秒杀活动
- 营销规则计算

### media-service - 多媒体服务
**职责**: 文件与媒体管理，负责图片、视频、文档存储
- 图片上传与处理（裁剪、缩放、水印）
- 视频上传与转码
- OSS 对象存储集成
- CDN 加速配置
- 文件访问权限控制

### message-service - 消息服务
**职责**: 消息通知，负责站内信、短信、邮件推送
- 站内信推送
- 短信发送（验证码、通知）
- 邮件发送（激活、通知）
- 消息模板管理
- 推送渠道配置

### user-action-service - 用户行为服务
**职责**: 用户行为数据采集与分析
- 浏览记录
- 收藏记录
- 足迹记录
- 搜索历史
- 行为数据分析

### commission-service - 佣金服务
**职责**: 分销与佣金管理
- 佣金计算规则
- 佣金结算
- 分销关系管理
- 提现申请处理

### cs-service - 客服服务
**职责**: 客户服务与售后支持
- 客服对话管理
- 工单创建与流转
- 常见问题管理
- 满意度评价

### common - 公共模块
**职责**: 跨服务共享的基础能力
- 统一错误处理
- 通用工具类
- 常量定义
- 公共配置
- 分布式 ID 生成器

## 通讯方式

结合`../protos`里的消息结构使用dubbo+grpc+protobuf进行通讯，借助`../Makefile`脚本生成胶水代码，再使用`@DubboReference`来跨服务调用方法体。protobuf的使用参考monorepo根目录的`../README.md`

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

每个微服务的数据库表结构定义在 `src/main/resources/db/schema.sql`。

### 服务目录结构

```
<service>/
├── src/
├── db/
│     ├── schema.sql      -- 表结构定义
│     └── migration/      -- 未来使用 Flyway / Liquibase 进行迁移
│           ├── V1__init.sql
│           ├── V2__add_index.sql
│           └── ...
```

### 当前已有 Schema 的服务

| 服务 | Schema 路径 |
|-----|-------------|
| cart-service | `cart-service/src/main/resources/db/schema.sql` |
| order-service | `order-service/src/main/resources/db/schema.sql` |
| product-service | `product-service/src/main/resources/db/schema.sql` |
| media-service | `media-service/src/main/resources/db/schema.sql` |
| user-service | `user-service/src/main/resources/db/schema.sql` |
| promotion-service | `promotion-service/src/main/resources/db/schema.sql` |
| user-action-service | `user-action-service/src/main/resources/db/schema.sql` |
| commission-service | `commission-service/src/main/resources/db/schema.sql` |
| payment-service | `payment-service/src/main/resources/db/schema.sql` |

### ID 生成策略

多微服务环境下，统一使用代码生成 ID（雪花算法），避免数据库自增冲突：

- **IdType.ASSIGN_ID**: 使用 MyBatis-Plus 雪花算法生成 ID
- **IdType.INPUT**: 手动设置 ID（如订单号等业务主键）

禁止使用 `IdType.AUTO`（数据库自增）。

## 支付模块配置

### 支付方式

| 支付方式 | SDK | 配置文件 |
|---------|-----|---------|
| 微信支付 | wechatpay-sdk-java | `payment-service/src/main/resources/application.yml` |
| 支付宝 | alipay-sdk-java | `payment-service/src/main/resources/application.yml` |
| Stripe | stripe-java | `payment-service/src/main/resources/application.yml` |

### 配置步骤

#### 1. 修改 application.yml

在 `payment-service/src/main/resources/application.yml` 中配置：

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

