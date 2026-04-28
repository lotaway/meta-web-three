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

