# Payment Service

法币兑换数字币微服务，提供完整的法币与数字资产兑换功能。

## 功能特性

### 1. 法币支付接入层
- 支持支付宝、微信支付、银行转账、Apple Pay、Google Pay
- 多种支付渠道的统一下单接口
- 支付回调处理和状态同步

### 2. 实时报价引擎
- 多交易所价格聚合（Binance、Coinbase、OKX）
- 加权平均价格计算
- 秒级价格更新
- 价格缓存机制

### 3. 订单撮合与兑换系统
- 支持法币购买数字币和数字币兑换法币
- 实时汇率锁定
- 滑点控制
- 自动撮合执行

### 4. 钱包与资金账户
- 热钱包和冷钱包管理
- 多币种支持（BTC、ETH、USDT、USDC）
- 区块链交易执行和确认

### 5. 订单状态同步
- 实时订单状态跟踪
- 支付状态同步
- 区块链交易确认

### 6. 合规与风控模块
- KYC级别验证
- 交易限额控制
- 异常行为检测
- 地址风险检查

## 技术架构

### 核心组件
- **ExchangeOrderService**: 兑换订单核心服务
- **PriceEngineService**: 价格引擎服务
- **RiskControlService**: 风控服务
- **PaymentService**: 支付服务
- **CryptoWalletService**: 数字钱包服务
- **ExternalPriceService**: 外部价格服务

### 数据模型
- **ExchangeOrder**: 兑换订单实体
- **CryptoPrice**: 加密货币价格实体
- **UserKYC**: 用户KYC实体

### 数据访问层
- **MyBatis Plus**: 使用MyBatis Plus进行数据访问
- **Mapper接口**: 继承BaseMapper，提供基础CRUD操作
- **XML映射**: 支持复杂SQL查询和结果映射
- **自动填充**: 支持创建时间和更新时间的自动填充

## API接口

### 兑换订单接口
```
POST /api/v1/exchange/orders          # 创建兑换订单
GET  /api/v1/exchange/orders/{orderNo} # 获取订单详情
GET  /api/v1/exchange/orders          # 获取用户订单列表
DELETE /api/v1/exchange/orders/{orderNo} # 取消订单
POST /api/v1/exchange/payment/callback # 支付回调
```

### 价格接口
```
GET /api/v1/prices/{symbol}           # 获取实时价格
GET /api/v1/prices/weighted/{base}/{quote} # 获取加权平均价格
GET /api/v1/prices/exchange-rate      # 计算兑换汇率
GET /api/v1/prices/{symbol}/change    # 获取价格变化
```

## 配置说明

### MyBatis Plus配置
```yaml
mybatis-plus:
  configuration:
    map-underscore-to-camel-case: true
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
  global-config:
    db-config:
      id-type: auto
      logic-delete-field: deleted
      logic-delete-value: 1
      logic-not-delete-value: 0
  mapper-locations: classpath*:/mapper/**/*.xml
```

### 支付配置
```yaml
payment:
  fiat:
    alipay:
      app-id: your_alipay_app_id
      private-key: your_alipay_private_key
      public-key: your_alipay_public_key
    wechat:
      app-id: your_wechat_app_id
      mch-id: your_wechat_mch_id
      api-key: your_wechat_api_key
```

### 风控配置
```yaml
payment:
  risk-control:
    single-limit:
      usd: 10000
      cny: 70000
    daily-limit:
      usd: 50000
      cny: 350000
```

### KYC配置
```yaml
payment:
  kyc:
    levels:
      l0: {name: "基础验证", limit: 1000}
      l1: {name: "身份验证", limit: 10000}
      l2: {name: "高级验证", limit: 100000}
      l3: {name: "企业验证", limit: 1000000}
```

## 部署说明

### 1. 数据库初始化
```sql
-- 执行 src/main/resources/db/init.sql
```

### 2. 环境变量配置
```bash
# 数据库配置
SPRING_DATASOURCE_URL=jdbc:mysql://localhost:3306/payment_service
SPRING_DATASOURCE_USERNAME=root
SPRING_DATASOURCE_PASSWORD=root

# Redis配置
SPRING_DATA_REDIS_HOST=localhost
SPRING_DATA_REDIS_PORT=6379

# 支付配置
PAYMENT_FIAT_ALIPAY_APP_ID=your_alipay_app_id
PAYMENT_FIAT_WECHAT_APP_ID=your_wechat_app_id
```

### 3. Docker部署
```bash
# 构建镜像
docker build -t payment-service .

# 运行容器
docker run -d -p 10086:10086 --name payment-service payment-service
```

## 业务流程

### 法币购买数字币流程
1. 用户发起购买请求
2. 系统验证KYC级别和风控规则
3. 获取实时价格并计算兑换数量
4. 创建兑换订单
5. 生成支付链接
6. 用户完成支付
7. 系统确认支付并执行数字资产转账
8. 订单完成

### 数字币兑换法币流程
1. 用户发起兑换请求
2. 系统验证KYC级别和风控规则
3. 获取实时价格并计算法币金额
4. 创建兑换订单
5. 用户转账数字资产到系统钱包
6. 系统确认到账并执行法币转账
7. 订单完成

## 监控和日志

### 关键指标
- 订单成功率
- 平均处理时间
- 价格更新频率
- 支付成功率
- 风控拦截率

### 日志级别
- DEBUG: 详细调试信息
- INFO: 业务操作日志
- WARN: 警告信息
- ERROR: 错误信息

## 安全考虑

### 数据安全
- 敏感数据加密存储
- 支付信息脱敏
- 访问日志记录

### 接口安全
- JWT身份验证
- 请求签名验证
- 频率限制
- IP白名单

### 风控安全
- 实时风控检测
- 异常行为识别
- 黑名单机制
- 限额控制

## 扩展性

### 支持新币种
1. 在配置中添加新币种
2. 实现对应的钱包服务
3. 添加价格源支持

### 支持新支付方式
1. 实现支付接口
2. 添加回调处理
3. 更新配置

### 支持新交易所
1. 实现价格获取接口
2. 添加权重配置
3. 更新价格聚合逻辑

## 数据访问层说明

### MyBatis Plus特性
- **BaseMapper**: 提供基础的CRUD操作
- **分页插件**: 支持分页查询
- **自动填充**: 自动处理创建时间和更新时间
- **逻辑删除**: 支持软删除
- **条件构造器**: 支持动态SQL构建

### Mapper接口
```java
@Mapper
public interface ExchangeOrderRepository extends BaseMapper<ExchangeOrder> {
    // 自定义查询方法
    @Select("SELECT * FROM exchange_orders WHERE order_no = #{orderNo}")
    ExchangeOrder findByOrderNo(@Param("orderNo") String orderNo);
}
```

### XML映射文件
- 位置：`src/main/resources/mapper/`
- 功能：定义复杂SQL查询和结果映射
- 命名空间：对应Mapper接口的完整类名 