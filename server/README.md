## AWS S3 Configuration

To use AWS S3, you need to configure the following properties in `.aws/credentials` file in the root directory of your project:

```bash
[default]
aws_access_key_id=s3_access_key
aws_secret_access_key=s3_secret_key
```

## Quartz Schedule Config

[Quartz SQL Scripts and Example](https://www.quartz-scheduler.org/downloads)

## Promotion Service

Spring Boot promotion microservice (coupon) is located at `promotion-service`.  
Default port: `10087`, Dubbo port: `20087`.  
Database schema: `promotion-service/src/main/resources/db/schema.sql`.

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
| commission-service | `commission-service/src/main/resources/db/schema.sql` |
| payment-service | `payment-service/src/main/resources/db/schema.sql` |

### ID 生成策略

多微服务环境下，统一使用代码生成 ID（雪花算法），避免数据库自增冲突：

- **IdType.ASSIGN_ID**: 使用 MyBatis-Plus 雪花算法生成 ID
- **IdType.INPUT**: 手动设置 ID（如订单号等业务主键）

禁止使用 `IdType.AUTO`（数据库自增）。

