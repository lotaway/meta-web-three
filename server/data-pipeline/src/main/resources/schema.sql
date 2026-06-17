-- ClickHouse Schema
CREATE DATABASE IF NOT EXISTS meta_web_analytics;

CREATE TABLE IF NOT EXISTS meta_web_analytics.order_analytics (
    event_id String, event_type String, order_id UInt64, user_id UInt64,
    total_amount Decimal(10, 2), status String, event_time DateTime,
    product_info String, payment_method String, merchant_id UInt64,
    processed_time DateTime, year_month String, day_of_week UInt8, hour_of_day UInt8
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, order_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS meta_web_analytics.inventory_analytics (
    event_id String, event_type String, product_id UInt64, product_name String,
    quantity Int32, available_qty UInt32, reserved_qty UInt32, warehouse_id String,
    event_time DateTime, operator String, remark String, processed_time DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, product_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS meta_web_analytics.user_behavior_analytics (
    event_id String, event_type String, user_id UInt64, session_id String,
    page_url String, referrer String, product_id UInt64, search_keyword String,
    category String, duration UInt32, device_type String, browser String,
    os String, ip_address String, event_time DateTime, extra_data String,
    processed_time DateTime, browser_family String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, user_id, session_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS meta_web_analytics.lineage_node (
    node_id String, name String, type String, system String,
    database_name String, table_name String, fields String,
    metadata String, created_at DateTime
) ENGINE = ReplacingMergeTree()
ORDER BY (node_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS meta_web_analytics.lineage_edge (
    edge_id String, source_node_id String, target_node_id String,
    edge_type String, transformation String, metadata String, created_at DateTime
) ENGINE = ReplacingMergeTree()
ORDER BY (edge_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS meta_web_analytics.column_lineage (
    source_node_id String, source_column String,
    target_node_id String, target_column String, transformation String
) ENGINE = ReplacingMergeTree()
ORDER BY (source_node_id, source_column, target_node_id, target_column)
SETTINGS index_granularity = 8192;

CREATE MATERIALIZED VIEW IF NOT EXISTS meta_web_analytics.mv_order_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (day, event_type, status, payment_method)
AS SELECT toDate(event_time) AS day, event_type, status, payment_method,
          count() AS order_count, sum(total_amount) AS total_amount,
          uniqExact(user_id) AS unique_users, uniqExact(order_id) AS unique_orders
   FROM meta_web_analytics.order_analytics
   GROUP BY day, event_type, status, payment_method;

CREATE MATERIALIZED VIEW IF NOT EXISTS meta_web_analytics.mv_user_behavior_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (day, event_type, device_type, browser_family)
AS SELECT toDate(event_time) AS day, event_type, device_type, browser_family, category,
          count() AS event_count, uniqExact(user_id) AS unique_users,
          uniqExact(session_id) AS unique_sessions, sum(duration) AS total_duration
   FROM meta_web_analytics.user_behavior_analytics
   GROUP BY day, event_type, device_type, browser_family, category;

CREATE MATERIALIZED VIEW IF NOT EXISTS meta_web_analytics.mv_inventory_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (day, event_type, warehouse_id)
AS SELECT toDate(event_time) AS day, event_type, warehouse_id,
          count() AS event_count, uniqExact(product_id) AS unique_products,
          sum(quantity) AS total_quantity
   FROM meta_web_analytics.inventory_analytics
   GROUP BY day, event_type, warehouse_id;
