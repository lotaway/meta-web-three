# 领域边界规范 (Domain Boundary Rules)

## 1. 概述

本文档定义 meta-web-three 项目中各业务领域的边界、数据所有权以及领域间通信规范。所有服务必须遵守本规范，以确保系统架构的可维护性和扩展性。

**核心原则**：每个领域的数据只能由该领域的核心服务负责修改，其他领域必须通过事件或 API 调用进行协作。

---

## 2. 领域划分与数据 Owner

### 2.1 领域定义

| 领域 | 英文 | 核心职责 |
|------|------|----------|
| 商城域 | mall-domain | 商品、订单、购物车、促销、支付、用户 |
| 供应链域 | supply-chain-domain | 库存、仓库、物流、采购、供应商 |
| 工厂域 | factory-domain | 生产制造、设备管理、MES |
| AI 域 | ai-domain | 风险评分、推荐系统、需求预测 |
| 区块链域 | blockchain-domain | 智能合约、NFT、链上结算 |
| 平台域 | platform-domain | 网关、媒体、消息、客服、佣金、用户行为 |

### 2.2 数据 Owner 规则

| 数据实体 | Owner 服务 | Owner 领域 | 说明 |
|----------|------------|------------|------|
| 商品 (Product) | product-service | mall-domain | 商品基本信息、规格、价格 |
| 订单 (Order) | order-service | mall-domain | 订单全生命周期 |
| 购物车 (Cart) | cart-service | mall-domain | 用户购物车 |
| 促销 (Promotion) | promotion-service | mall-domain | 促销活动、优惠券 |
| 支付 (Payment) | payment-service | mall-domain | 支付流水、退款 |
| 用户 (User) | user-service | mall-domain | 用户基本信息、认证 |
| 库存 (Inventory) | inventory-service | supply-chain-domain | SKU 库存数量、预留 |
| 仓库 (Warehouse) | warehouse-service | supply-chain-domain | 仓库信息、库位 |
| 物流 (Logistics) | logistics-service | supply-chain-domain | 物流轨迹、承运商 |
| 采购 (Procurement) | procurement-service | supply-chain-domain | 采购单、供应商订单 |
| 供应商 (Supplier) | supplier-service | supply-chain-domain | 供应商信息、合作关系 |
| 佣金 (Commission) | commission-service | platform-domain | 分销佣金计算 |
| 媒体 (Media) | media-service | platform-domain | 图片、视频存储 |
| 消息 (Message) | message-service | platform-domain | 站内信、通知 |
| 客服 (CS) | cs-service | platform-domain | 客服工单 |
| 用户行为 (UserAction) | user-action-service | platform-domain | 行为埋点、分析 |

---

## 3. 领域间通信规范

### 3.1 允许的通信方式

1. **事件驱动（Event-Driven）** - 推荐方式
   - 使用 Kafka 进行领域间通信
   - 发布方不关心消费方是谁
   - 解耦领域间的强依赖

2. **同步 API 调用（API Call）** - 仅限于查询场景
   - GET 请求查询其他领域数据
   - 需要显式声明依赖关系
   - 禁止在同步调用中修改对方数据

### 3.2 禁止的通信方式

❌ **直接跨域写库**  
任何服务禁止直接修改非本领域的数据表。例如：
- order-service 禁止直接更新 inventory 表
- product-service 禁止直接更新 order 表

❌ **跨域事务**  
禁止使用分布式事务直接操作多个领域的数据源。

❌ **直接调用其他服务的写方法**  
禁止通过 HTTP/gRPC 直接调用其他领域服务的写操作（POST/PUT/DELETE）。

### 3.3 事件通信示例

```java
// 订单创建后，发布事件
@PostMapping("/orders")
public ResponseEntity<Order> createOrder(@RequestBody OrderRequest request) {
    Order order = orderService.create(request);
    
    // 发布事件，通知其他领域
    eventPublisher.publish(OrderCreatedEvent.builder()
        .orderId(order.getId())
        .userId(order.getUserId())
        .items(order.getItems())
        .totalAmount(order.getTotalAmount())
        .build());
    
    return ResponseEntity.ok(order);
}
```

```java
// 库存服务订阅事件
@KafkaListener(topics = "order.created")
public void handleOrderCreated(OrderCreatedEvent event) {
    // 预留库存
    inventoryService.reserve(event.getOrderId(), event.getItems());
}
```

---

## 4. CI 跨域检查

### 4.1 检查规则

在 GitHub Actions / GitLab CI 中添加跨域检查：

1. **数据库访问检查**
   - 扫描各服务的数据库访问代码
   - 检查是否访问了非本领域的表

2. **依赖关系检查**
   - 检查 pom.xml / package.json 中的依赖
   - 禁止直接依赖其他领域的核心服务

3. **SQL 审核**
   - 使用 Flyway / Liquibase 审核迁移 SQL
   - 检查是否修改了非本领域的表

### 4.2 检查脚本示例

```bash
#!/bin/bash
# check_domain_boundary.sh

SERVICE=$1
ALLOWED_TABLES=$(cat config/domain/${SERVICE}_allowed_tables)

# 检查 SQL 迁移文件
for sql_file in $(find . -name "V*.sql"); do
    tables=$(grep -oP 'INSERT INTO \K\w+|UPDATE \K\w+|DELETE FROM \K\w+' "$sql_file" | sort -u)
    for table in $tables; do
        if ! echo "$ALLOWED_TABLES" | grep -qw "$table"; then
            echo "ERROR: $SERVICE attempted to access forbidden table: $table"
            exit 1
        fi
    done
done
```

---

## 5. 违反规则的处理

| 违规类型 | 处理方式 |
|----------|----------|
| 直接跨域写库 | Code Review 拒绝 + 修复任务 |
| 跨域同步写操作 | CI 失败 + 重构为事件驱动 |
| 未授权的数据访问 | 安全漏洞级别 Bug |

---

## 6. 各领域核心事件定义

### 6.1 mall-domain 事件

| 事件名 | 发布方 | 订阅方 |
|--------|--------|--------|
| `order.created` | order-service | inventory-service, notification-service |
| `order.completed` | order-service | promotion-service, commission-service |
| `order.cancelled` | order-service | inventory-service, payment-service |
| `payment.succeeded` | payment-service | order-service, inventory-service |
| `payment.failed` | payment-service | order-service |

### 6.2 supply-chain-domain 事件

| 事件名 | 发布方 | 订阅方 |
|--------|--------|--------|
| `inventory.reserved` | inventory-service | warehouse-service |
| `inventory.released` | inventory-service | - |
| `inventory.low_stock` | inventory-service | procurement-service |
| `shipment.created` | logistics-service | order-service |
| `shipment.delivered` | logistics-service | order-service |

---

## 7. 配置示例

### 7.1 服务允许表配置

```yaml
# config/domain/order-service.yaml
owner_domain: mall-domain
allowed_tables:
  - orders
  - order_items
  - order_logs
readonly_tables:
  - products
  - inventory
```

```yaml
# config/domain/inventory-service.yaml
owner_domain: supply-chain-domain
allowed_tables:
  - inventory
  - inventory_reserved
  - inventory_logs
readonly_tables:
  - orders
```

---

## 8. 附录

### 8.1 术语解释

- **Owner（数据拥有者）**：唯一有权限修改特定数据的服务
- **Domain（领域）**：业务边界的逻辑分组
- **Event-Driven（事件驱动）**：通过消息队列解耦服务间通信

### 8.2 相关文档

- [CODE_PRICEPLES](./CODE_PINCEPLES) - 后端代码规范
- [BLOCK_CHAIN_CONTRACT_PRICEPLES](./CODE_PINCEPLES) - 区块链合约规范
- [事件总线建设文档](./shared/event-sdk/README.md)