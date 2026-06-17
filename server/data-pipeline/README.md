# Data Pipeline Service

Real-time ETL (Extract, Transform, Load) pipeline for processing order, inventory, and user behavior data in the meta-web-three e-commerce platform.

## Overview

This service consumes events from Kafka topics, transforms them into analytics-ready format, and loads them into ClickHouse for real-time OLAP queries.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Order Service  │────▶│    Kafka Topic   │────▶│   Data          │
│  Inventory      │     │  - order-events  │     │   Pipeline      │
│  User Service   │     │  - inventory     │     │   Service       │
│                 │     │  - user-behavior │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
                                                    ┌─────────────────┐
                                                    │   ClickHouse    │
                                                    │   Analytics     │
                                                    │   Database      │
                                                    └─────────────────┘
                                                           │
                                                           ▼
                                                    ┌─────────────────┐
                                                    │   Real-time     │
                                                    │   Dashboard     │
                                                    │   & Reports     │
                                                    └─────────────────┘
```

## Features

- **Real-time Event Processing**: Consumes order, inventory, and user behavior events from Kafka
- **Data Transformation**: Converts raw events into analytics-ready format with derived fields
- **ClickHouse Integration**: High-performance OLAP database for analytics queries
- **Batch Processing**: Configurable batch size and interval for efficient processing
- **Error Handling**: Dead letter queue support for failed events
- **Monitoring**: Spring Boot Actuator endpoints for health checks and metrics

## Event Types

### Order Events
- **Event Types**: CREATE, UPDATE, PAY, CANCEL, COMPLETE
- **Fields**: orderId, userId, totalAmount, status, eventTime, productInfo, paymentMethod, merchantId
- **Analytics Fields**: yearMonth, dayOfWeek, hourOfDay (derived)

### Inventory Events
- **Event Types**: STOCK_IN, STOCK_OUT, ADJUST, ALERT
- **Fields**: productId, productName, quantity, availableQty, reservedQty, warehouseId, eventTime, operator

### User Behavior Events
- **Event Types**: PAGE_VIEW, PRODUCT_VIEW, ADD_TO_CART, SEARCH, CLICK, PURCHASE
- **Fields**: userId, sessionId, pageUrl, referrer, productId, searchKeyword, category, duration, deviceType, browser, os, ipAddress

## Configuration

### Application Properties

```yaml
server:
  port: 10122

spring:
  application:
    name: data-pipeline
  
kafka:
  bootstrap-servers: localhost:9092
  consumer:
    group-id: data-pipeline-group
    auto-offset-reset: earliest

clickhouse:
  url: jdbc:clickhouse://localhost:8123/meta_web_analytics
  username: default
  password: ""

etl:
  batch:
    size: 100
    interval: 5000
  topics:
    order: meta-web-order-events
    inventory: meta-web-inventory-events
    user-behavior: meta-web-user-behavior-events
```

## ClickHouse Schema

### Order Analytics Table
```sql
CREATE TABLE meta_web_analytics.order_analytics (
    event_id String,
    event_type String,
    order_id UInt64,
    user_id UInt64,
    total_amount Decimal(10, 2),
    status String,
    event_time DateTime,
    product_info String,
    payment_method String,
    merchant_id UInt64,
    processed_time DateTime,
    year_month String,
    day_of_week UInt8,
    hour_of_day UInt8
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, order_id)
```

### Inventory Analytics Table
```sql
CREATE TABLE meta_web_analytics.inventory_analytics (
    event_id String,
    event_type String,
    product_id UInt64,
    product_name String,
    quantity Int32,
    available_qty UInt32,
    reserved_qty UInt32,
    warehouse_id String,
    event_time DateTime,
    operator String,
    remark String,
    processed_time DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, product_id)
```

### User Behavior Analytics Table
```sql
CREATE TABLE meta_web_analytics.user_behavior_analytics (
    event_id String,
    event_type String,
    user_id UInt64,
    session_id String,
    page_url String,
    referrer String,
    product_id UInt64,
    search_keyword String,
    category String,
    duration UInt32,
    device_type String,
    browser String,
    os String,
    ip_address String,
    event_time DateTime,
    extra_data String,
    processed_time DateTime,
    browser_family String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, user_id, session_id)
```

## API Endpoints

### Health Check
```
GET /actuator/health
```

### Metrics
```
GET /actuator/metrics
GET /actuator/prometheus
```

## Usage

### 1. Start Kafka and ClickHouse

```bash
# Start Kafka
docker-compose -f infra/kafka/docker-compose.yml up -d

# Start ClickHouse
docker run -d --name clickhouse \
  -p 8123:8123 -p 9000:9000 \
  clickhouse/clickhouse-server:latest
```

### 2. Create Kafka Topics

```bash
# Create order events topic
kafka-topics --create --topic meta-web-order-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 1

# Create inventory events topic
kafka-topics --create --topic meta-web-inventory-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 1

# Create user behavior events topic
kafka-topics --create --topic meta-web-user-behavior-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 1
```

### 3. Run Data Pipeline Service

```bash
cd server/data-pipeline
mvn spring-boot:run
```

### 4. Produce Test Events

```bash
# Produce order event
kafka-console-producer --topic meta-web-order-events \
  --bootstrap-server localhost:9092 < order-event.json

# Produce inventory event
kafka-console-producer --topic meta-web-inventory-events \
  --bootstrap-server localhost:9092 < inventory-event.json

# Produce user behavior event
kafka-console-producer --topic meta-web-user-behavior-events \
  --bootstrap-server localhost:9092 < user-behavior-event.json
```

## Sample Event JSON

### Order Event
```json
{
  "eventId": "evt_order_001",
  "eventType": "CREATE",
  "orderId": 10001,
  "userId": 5001,
  "totalAmount": 299.99,
  "status": "PENDING",
  "eventTime": "2026-06-03T20:30:00",
  "productInfo": "[{\"productId\":101,\"quantity\":2}]",
  "paymentMethod": "ALIPAY",
  "merchantId": 201
}
```

### Inventory Event
```json
{
  "eventId": "evt_inv_001",
  "eventType": "STOCK_OUT",
  "productId": 101,
  "productName": "Smartphone X",
  "quantity": -5,
  "availableQty": 95,
  "reservedQty": 10,
  "warehouseId": "WH001",
  "eventTime": "2026-06-03T20:30:00",
  "operator": "admin",
  "remark": "Order #10001"
}
```

### User Behavior Event
```json
{
  "eventId": "evt_usr_001",
  "eventType": "PRODUCT_VIEW",
  "userId": 5001,
  "sessionId": "sess_abc123",
  "pageUrl": "/product/101",
  "referrer": "/category/electronics",
  "productId": 101,
  "searchKeyword": null,
  "category": "Electronics",
  "duration": 120,
  "deviceType": "MOBILE",
  "browser": "Chrome",
  "os": "iOS",
  "ipAddress": "192.168.1.100",
  "eventTime": "2026-06-03T20:30:00",
  "extraData": "{\"color\":\"black\",\"storage\":\"256GB\"}"
}
```

## Monitoring Queries (ClickHouse)

### Daily Order Count
```sql
SELECT 
  toDate(event_time) as date,
  event_type,
  count() as count,
  sum(total_amount) as total_amount
FROM meta_web_analytics.order_analytics
WHERE event_time >= today() - 7
GROUP BY date, event_type
ORDER BY date DESC, event_type;
```

### Top Viewed Products (Last 24h)
```sql
SELECT 
  product_id,
  count() as view_count
FROM meta_web_analytics.user_behavior_analytics
WHERE event_type = 'PRODUCT_VIEW'
  AND event_time >= now() - INTERVAL 24 HOUR
GROUP BY product_id
ORDER BY view_count DESC
LIMIT 10;
```

### Inventory Alerts
```sql
SELECT 
  product_id,
  product_name,
  available_qty,
  event_time
FROM meta_web_analytics.inventory_analytics
WHERE event_type = 'ALERT'
  AND event_time >= now() - INTERVAL 1 HOUR
ORDER BY event_time DESC;
```

## Dependencies

- Spring Boot 3.5.4
- Spring Kafka
- ClickHouse JDBC
- RocketMQ Spring Boot Starter
- Lombok
- Jackson

## TODO

- [ ] Add ClickHouse or Druid real-time OLAP query support
- [ ] Build real-time dashboard data push service (WebSocket)
- [ ] Implement data lineage tracking (Data Lineage)

## License

MIT License
