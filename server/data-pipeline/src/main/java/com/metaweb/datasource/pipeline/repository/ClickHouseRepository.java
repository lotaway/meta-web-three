package com.metaweb.datasource.pipeline.repository;

import com.metaweb.datasource.pipeline.service.EtlService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;

import javax.annotation.PostConstruct;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Repository
public class ClickHouseRepository {

    private static final String DATABASE_NAME = "meta_web_analytics";
    private static final int INDEX_GRANULARITY = 8192;
    private static final String TOTAL_AMOUNT_DECIMAL = "Decimal(10, 2)";

    @Value("${clickhouse.url}")
    private String clickhouseUrl;

    @Value("${clickhouse.username}")
    private String username;

    @Value("${clickhouse.password}")
    private String password;

    private Connection connection;

    @PostConstruct
    public void init() {
        try {
            Class.forName("com.clickhouse.jdbc.ClickHouseDriver");
            log.info("ClickHouse driver loaded successfully");
            initDatabase();
        } catch (ClassNotFoundException e) {
            log.error("Failed to load ClickHouse driver", e);
            throw new RuntimeException("ClickHouse driver not found", e);
        }
    }

    private void initDatabase() {
        try {
            String baseUrl = clickhouseUrl.substring(0, clickhouseUrl.lastIndexOf("/"));
            try (Connection conn = DriverManager.getConnection(baseUrl, username, password)) {
                Statement stmt = conn.createStatement();
                stmt.execute("CREATE DATABASE IF NOT EXISTS " + DATABASE_NAME);
                log.info("Database {} ensured", DATABASE_NAME);
            }
            Connection conn = getConnection();
            Statement stmt = conn.createStatement();
            createOrderAnalyticsTable(stmt);
            createInventoryAnalyticsTable(stmt);
            createUserBehaviorAnalyticsTable(stmt);
            log.info("ClickHouse tables initialized successfully");
        } catch (SQLException e) {
            log.error("Failed to initialize ClickHouse database", e);
            throw new RuntimeException("ClickHouse initialization failed", e);
        }
    }

    private void createOrderAnalyticsTable(Statement stmt) throws SQLException {
        stmt.execute("""
            CREATE TABLE IF NOT EXISTS """ + DATABASE_NAME + """.order_analytics (
                event_id String, event_type String, order_id UInt64, user_id UInt64,
                total_amount """ + TOTAL_AMOUNT_DECIMAL + """, status String, event_time DateTime,
                product_info String, payment_method String, merchant_id UInt64,
                processed_time DateTime, year_month String, day_of_week UInt8, hour_of_day UInt8
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (event_time, order_id)
            SETTINGS index_granularity = """ + INDEX_GRANULARITY + """
        """);
    }

    private void createInventoryAnalyticsTable(Statement stmt) throws SQLException {
        stmt.execute("""
            CREATE TABLE IF NOT EXISTS """ + DATABASE_NAME + """.inventory_analytics (
                event_id String, event_type String, product_id UInt64, product_name String,
                quantity Int32, available_qty UInt32, reserved_qty UInt32, warehouse_id String,
                event_time DateTime, operator String, remark String, processed_time DateTime
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (event_time, product_id)
            SETTINGS index_granularity = """ + INDEX_GRANULARITY + """
        """);
    }

    private void createUserBehaviorAnalyticsTable(Statement stmt) throws SQLException {
        stmt.execute("""
            CREATE TABLE IF NOT EXISTS """ + DATABASE_NAME + """.user_behavior_analytics (
                event_id String, event_type String, user_id UInt64, session_id String,
                page_url String, referrer String, product_id UInt64, search_keyword String,
                category String, duration UInt32, device_type String, browser String,
                os String, ip_address String, event_time DateTime, extra_data String,
                processed_time DateTime, browser_family String
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (event_time, user_id, session_id)
            SETTINGS index_granularity = """ + INDEX_GRANULARITY + """
        """);
    }

    private Connection getConnection() throws SQLException {
        if (connection == null || connection.isClosed()) {
            connection = DriverManager.getConnection(clickhouseUrl, username, password);
        }
        return connection;
    }

    public void insertOrderAnalytics(EtlService.OrderAnalytics analytics) {
        String sql = "INSERT INTO " + DATABASE_NAME + ".order_analytics " +
            "(event_id, event_type, order_id, user_id, total_amount, status, " +
            "event_time, product_info, payment_method, merchant_id, " +
            "processed_time, year_month, day_of_week, hour_of_day) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        try (PreparedStatement pstmt = getConnection().prepareStatement(sql)) {
            setOrderParams(pstmt, analytics);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            log.error("Failed to insert order analytics: {}", analytics.getOrderId(), e);
            throw new RuntimeException("ClickHouse insert failed", e);
        }
    }

    private void setOrderParams(PreparedStatement pstmt, EtlService.OrderAnalytics analytics) throws SQLException {
        pstmt.setString(1, analytics.getEventId());
        pstmt.setString(2, analytics.getEventType());
        pstmt.setLong(3, analytics.getOrderId());
        pstmt.setLong(4, analytics.getUserId());
        pstmt.setBigDecimal(5, analytics.getTotalAmount());
        pstmt.setString(6, analytics.getStatus());
        pstmt.setTimestamp(7, Timestamp.valueOf(analytics.getEventTime()));
        pstmt.setString(8, analytics.getProductInfo());
        pstmt.setString(9, analytics.getPaymentMethod());
        if (analytics.getMerchantId() != null) {
            pstmt.setLong(10, analytics.getMerchantId());
        } else {
            pstmt.setNull(10, Types.BIGINT);
        }
        pstmt.setTimestamp(11, Timestamp.valueOf(analytics.getProcessedTime()));
        pstmt.setString(12, analytics.getYearMonth());
        pstmt.setInt(13, analytics.getDayOfWeek());
        pstmt.setInt(14, analytics.getHourOfDay());
    }

    public void insertInventoryAnalytics(EtlService.InventoryAnalytics analytics) {
        String sql = "INSERT INTO " + DATABASE_NAME + ".inventory_analytics " +
            "(event_id, event_type, product_id, product_name, quantity, " +
            "available_qty, reserved_qty, warehouse_id, event_time, " +
            "operator, remark, processed_time) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        try (PreparedStatement pstmt = getConnection().prepareStatement(sql)) {
            setInventoryParams(pstmt, analytics);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            log.error("Failed to insert inventory analytics: {}", analytics.getProductId(), e);
            throw new RuntimeException("ClickHouse insert failed", e);
        }
    }

    private void setInventoryParams(PreparedStatement pstmt, EtlService.InventoryAnalytics analytics) throws SQLException {
        pstmt.setString(1, analytics.getEventId());
        pstmt.setString(2, analytics.getEventType());
        pstmt.setLong(3, analytics.getProductId());
        pstmt.setString(4, analytics.getProductName());
        pstmt.setInt(5, analytics.getQuantity());
        pstmt.setInt(6, analytics.getAvailableQty());
        pstmt.setInt(7, analytics.getReservedQty());
        pstmt.setString(8, analytics.getWarehouseId());
        pstmt.setTimestamp(9, Timestamp.valueOf(analytics.getEventTime()));
        pstmt.setString(10, analytics.getOperator());
        pstmt.setString(11, analytics.getRemark());
        pstmt.setTimestamp(12, Timestamp.valueOf(analytics.getProcessedTime()));
    }

    public void insertUserBehaviorAnalytics(EtlService.UserBehaviorAnalytics analytics) {
        String sql = "INSERT INTO " + DATABASE_NAME + ".user_behavior_analytics " +
            "(event_id, event_type, user_id, session_id, page_url, " +
            "referrer, product_id, search_keyword, category, duration, " +
            "device_type, browser, os, ip_address, event_time, " +
            "extra_data, processed_time, browser_family) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        try (PreparedStatement pstmt = getConnection().prepareStatement(sql)) {
            setUserBehaviorParams(pstmt, analytics);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            log.error("Failed to insert user behavior analytics: {}", analytics.getEventId(), e);
            throw new RuntimeException("ClickHouse insert failed", e);
        }
    }

    private void setUserBehaviorParams(PreparedStatement pstmt, EtlService.UserBehaviorAnalytics analytics) throws SQLException {
        pstmt.setString(1, analytics.getEventId());
        pstmt.setString(2, analytics.getEventType());
        if (analytics.getUserId() != null) {
            pstmt.setLong(3, analytics.getUserId());
        } else {
            pstmt.setNull(3, Types.BIGINT);
        }
        pstmt.setString(4, analytics.getSessionId());
        pstmt.setString(5, analytics.getPageUrl());
        pstmt.setString(6, analytics.getReferrer());
        if (analytics.getProductId() != null) {
            pstmt.setLong(7, analytics.getProductId());
        } else {
            pstmt.setNull(7, Types.BIGINT);
        }
        pstmt.setString(8, analytics.getSearchKeyword());
        pstmt.setString(9, analytics.getCategory());
        if (analytics.getDuration() != null) {
            pstmt.setInt(10, analytics.getDuration());
        } else {
            pstmt.setNull(10, Types.INTEGER);
        }
        pstmt.setString(11, analytics.getDeviceType());
        pstmt.setString(12, analytics.getBrowser());
        pstmt.setString(13, analytics.getOs());
        pstmt.setString(14, analytics.getIpAddress());
        pstmt.setTimestamp(15, Timestamp.valueOf(analytics.getEventTime()));
        pstmt.setString(16, analytics.getExtraData());
        pstmt.setTimestamp(17, Timestamp.valueOf(analytics.getProcessedTime()));
        pstmt.setString(18, analytics.getBrowserFamily());
    }

    public void close() {
        if (connection != null) {
            try {
                if (!connection.isClosed()) {
                    connection.close();
                }
            } catch (SQLException e) {
                log.error("Failed to close ClickHouse connection", e);
            }
        }
    }
}
