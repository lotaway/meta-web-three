package com.metaweb.datasource.pipeline.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metaweb.datasource.pipeline.websocket.DashboardWebSocketHandler;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Slf4j
@Service
public class DashboardPushService {

    private static final long DASHBOARD_PUSH_INTERVAL_MS = 5000;

    private static final String ORDER_LAST_HOUR_SQL = """
        SELECT count() AS order_count, sum(total_amount) AS total_amount, uniqExact(user_id) AS unique_users
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= now() - INTERVAL 1 HOUR AND event_type = 'CREATE'
        """;

    private static final String ORDER_TODAY_SQL = """
        SELECT count() AS order_count, sum(total_amount) AS total_amount, uniqExact(user_id) AS unique_users, avg(total_amount) AS avg_order_amount
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= today() AND event_type = 'CREATE'
        """;

    private static final String STATUS_SQL = """
        SELECT status, count() AS count
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= today()
        GROUP BY status ORDER BY count DESC
        """;

    private static final String ACTIVE_USERS_SQL = """
        SELECT uniqExact(user_id) AS active_users
        FROM meta_web_analytics.user_behavior_analytics
        WHERE event_time >= now() - INTERVAL 5 MINUTE
        """;

    private static final String TOP_PRODUCTS_SQL = """
        SELECT product_id, count() AS view_count
        FROM meta_web_analytics.user_behavior_analytics
        WHERE event_time >= now() - INTERVAL 1 HOUR AND event_type = 'PRODUCT_VIEW' AND product_id > 0
        GROUP BY product_id ORDER BY view_count DESC LIMIT 5
        """;

    private static final String REVENUE_BY_HOUR_SQL = """
        SELECT toHour(event_time) AS hour, sum(total_amount) AS revenue
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= today() AND event_type = 'CREATE'
        GROUP BY hour ORDER BY hour
        """;

    private static final String INVENTORY_ALERT_SQL = """
        SELECT count() AS alert_count, uniqExact(product_id) AS affected_products
        FROM meta_web_analytics.inventory_analytics
        WHERE event_time >= now() - INTERVAL 1 HOUR AND event_type = 'ALERT'
        """;

    private static final String DEVICE_SQL = """
        SELECT device_type, count() AS count
        FROM meta_web_analytics.user_behavior_analytics
        WHERE event_time >= now() - INTERVAL 1 HOUR
        GROUP BY device_type ORDER BY count DESC
        """;

    @Autowired
    private DashboardWebSocketHandler webSocketHandler;

    @Autowired
    @Qualifier("clickHouseJdbcTemplate")
    private JdbcTemplate clickHouseJdbcTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    @Scheduled(fixedRate = DASHBOARD_PUSH_INTERVAL_MS)
    public void pushDashboardMetrics() {
        if (webSocketHandler.getActiveSessionCount() == 0) {
            return;
        }
        try {
            Map<String, Object> payload = collectDashboardMetrics();
            payload.put("timestamp", LocalDateTime.now().format(FMT));
            payload.put("type", "METRICS");
            String json = objectMapper.writeValueAsString(payload);
            webSocketHandler.broadcast(json);
        } catch (Exception e) {
            log.error("Failed to push dashboard metrics: {}", e.getMessage());
        }
    }

    public void pushOrderAlert(Map<String, Object> orderEvent) {
        try {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "ORDER_ALERT");
            payload.put("data", orderEvent);
            payload.put("timestamp", LocalDateTime.now().format(FMT));
            String json = objectMapper.writeValueAsString(payload);
            webSocketHandler.broadcast(json);
        } catch (Exception e) {
            log.error("Failed to push order alert: {}", e.getMessage());
        }
    }

    public void pushInventoryAlert(Map<String, Object> inventoryEvent) {
        try {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "INVENTORY_ALERT");
            payload.put("data", inventoryEvent);
            payload.put("timestamp", LocalDateTime.now().format(FMT));
            String json = objectMapper.writeValueAsString(payload);
            webSocketHandler.broadcast(json);
        } catch (Exception e) {
            log.error("Failed to push inventory alert: {}", e.getMessage());
        }
    }

    private Map<String, Object> collectDashboardMetrics() {
        Map<String, Object> metrics = new LinkedHashMap<>();
        try {
            metrics.put("ordersLastHour", querySafe(ORDER_LAST_HOUR_SQL));
            metrics.put("ordersToday", querySafe(ORDER_TODAY_SQL));
            metrics.put("orderStatusDistribution", queryListSafe(STATUS_SQL));
            metrics.put("activeUsers", querySafe(ACTIVE_USERS_SQL));
            metrics.put("topViewedProducts", queryListSafe(TOP_PRODUCTS_SQL));
            metrics.put("revenueByHour", queryListSafe(REVENUE_BY_HOUR_SQL));
            metrics.put("inventoryAlerts", querySafe(INVENTORY_ALERT_SQL));
            metrics.put("deviceBreakdown", queryListSafe(DEVICE_SQL));
        } catch (Exception e) {
            log.error("Error collecting dashboard metrics: {}", e.getMessage());
            metrics.put("error", e.getMessage());
        }
        return metrics;
    }

    private Map<String, Object> querySafe(String sql) {
        try {
            return clickHouseJdbcTemplate.queryForMap(sql);
        } catch (Exception e) {
            log.warn("ClickHouse query failed: {}", e.getMessage());
            return Map.of();
        }
    }

    private List<Map<String, Object>> queryListSafe(String sql) {
        try {
            return clickHouseJdbcTemplate.queryForList(sql);
        } catch (Exception e) {
            log.warn("ClickHouse query failed: {}", e.getMessage());
            return List.of();
        }
    }
}
