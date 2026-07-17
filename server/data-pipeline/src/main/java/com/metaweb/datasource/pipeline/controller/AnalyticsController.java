package com.metaweb.datasource.pipeline.controller;

import com.metaweb.datasource.pipeline.repository.ClickHouseRepository;
import com.metaweb.datasource.pipeline.service.OlapQueryModels;
import com.metaweb.datasource.pipeline.service.OlapQueryService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.jdbc.core.JdbcTemplate;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Slf4j
@RestController
@RequestMapping("/api/analytics")
public class AnalyticsController {

    private static final String TABLE_ORDER = "meta_web_analytics.order_analytics";
    private static final String TABLE_INVENTORY = "meta_web_analytics.inventory_analytics";
    private static final String TABLE_USER_BEHAVIOR = "meta_web_analytics.user_behavior_analytics";

    private static final String ORDER_COLUMNS = "event_id, event_type, order_id, user_id, total_amount, status, event_time, year_month, day_of_week, hour_of_day";
    private static final String INVENTORY_COLUMNS = "event_id, event_type, product_id, product_name, quantity, available_qty, reserved_qty, warehouse_id, event_time";
    private static final String USER_BEHAVIOR_COLUMNS = "event_id, event_type, user_id, session_id, page_url, product_id, device_type, browser_family, event_time";

    private static final String ORDER_STATS_SQL = """
        SELECT COUNT(*) as total_orders, SUM(total_amount) as total_amount, COUNT(DISTINCT user_id) as unique_users
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= now() - INTERVAL ? HOUR AND event_type = 'CREATE'
        """;

    private static final String INVENTORY_STATS_SQL = """
        SELECT COUNT(*) as total_alerts, COUNT(DISTINCT product_id) as affected_products
        FROM meta_web_analytics.inventory_analytics
        WHERE event_time >= now() - INTERVAL ? HOUR AND event_type = 'ALERT'
        """;

    private static final String BEHAVIOR_STATS_SQL = """
        SELECT COUNT(*) as total_events, COUNT(DISTINCT user_id) as active_users, COUNT(DISTINCT session_id) as total_sessions
        FROM meta_web_analytics.user_behavior_analytics
        WHERE event_time >= now() - INTERVAL ? HOUR
        """;

    private static final String TOP_VIEWED_SQL = """
        SELECT product_id, COUNT(*) as view_count
        FROM meta_web_analytics.user_behavior_analytics
        WHERE event_time >= now() - INTERVAL ? HOUR AND event_type = 'PRODUCT_VIEW'
        GROUP BY product_id ORDER BY view_count DESC LIMIT 10
        """;

    private static final String ORDERS_BY_HOUR_SQL = """
        SELECT toHour(event_time) as hour, COUNT(*) as order_count
        FROM meta_web_analytics.order_analytics
        WHERE event_time >= now() - INTERVAL ? HOUR AND event_type = 'CREATE'
        GROUP BY hour ORDER BY hour
        """;

    @Autowired
    private ClickHouseRepository clickHouseRepository;

    @Autowired
    @Qualifier("clickHouseJdbcTemplate")
    private JdbcTemplate clickHouseJdbcTemplate;

    @Autowired
    private OlapQueryService olapQueryService;

    @GetMapping("/orders")
    public ResponseEntity<Map<String, Object>> getOrderAnalytics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) String eventType,
            @RequestParam(required = false) Long userId,
            @RequestParam(defaultValue = "0") Integer offset,
            @RequestParam(defaultValue = "20") Integer limit) {
        try {
            List<Object> params = new ArrayList<>();
            String sql = buildSelectSql(ORDER_COLUMNS, TABLE_ORDER, params, startTime, endTime, eventType, userId, null, offset, limit);
            List<Map<String, Object>> results = clickHouseJdbcTemplate.queryForList(sql, params.toArray());
            List<Object> countParams = new ArrayList<>();
            String countSql = buildCountSql(TABLE_ORDER, countParams, startTime, endTime, eventType, userId, null);
            Integer total = clickHouseJdbcTemplate.queryForObject(countSql, Integer.class, countParams.toArray());
            return ResponseEntity.ok(buildPageResponse(results, total, offset, limit));
        } catch (Exception e) {
            log.error("Failed to query order analytics", e);
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/inventory")
    public ResponseEntity<Map<String, Object>> getInventoryAnalytics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) String eventType,
            @RequestParam(required = false) Long productId,
            @RequestParam(defaultValue = "0") Integer offset,
            @RequestParam(defaultValue = "20") Integer limit) {
        try {
            List<Object> params = new ArrayList<>();
            String sql = buildSelectSql(INVENTORY_COLUMNS, TABLE_INVENTORY, params, startTime, endTime, eventType, null, productId, offset, limit);
            List<Map<String, Object>> results = clickHouseJdbcTemplate.queryForList(sql, params.toArray());
            List<Object> countParams = new ArrayList<>();
            String countSql = buildCountSql(TABLE_INVENTORY, countParams, startTime, endTime, eventType, null, productId);
            Integer total = clickHouseJdbcTemplate.queryForObject(countSql, Integer.class, countParams.toArray());
            return ResponseEntity.ok(buildPageResponse(results, total, offset, limit));
        } catch (Exception e) {
            log.error("Failed to query inventory analytics", e);
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/user-behavior")
    public ResponseEntity<Map<String, Object>> getUserBehaviorAnalytics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) String eventType,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long productId,
            @RequestParam(defaultValue = "0") Integer offset,
            @RequestParam(defaultValue = "20") Integer limit) {
        try {
            return executeUserBehaviorQuery(startTime, endTime, eventType, userId, productId, offset, limit);
        } catch (Exception e) {
            log.error("Failed to query user behavior analytics", e);
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    private ResponseEntity<Map<String, Object>> executeUserBehaviorQuery(
            LocalDateTime startTime, LocalDateTime endTime, String eventType,
            Long userId, Long productId, int offset, int limit) {
        List<Object> params = new ArrayList<>();
        String sql = buildSelectSql(USER_BEHAVIOR_COLUMNS, TABLE_USER_BEHAVIOR, params, startTime, endTime, eventType, userId, productId, offset, limit);
        List<Map<String, Object>> results = clickHouseJdbcTemplate.queryForList(sql, params.toArray());
        List<Object> countParams = new ArrayList<>();
        String countSql = buildCountSql(TABLE_USER_BEHAVIOR, countParams, startTime, endTime, eventType, userId, productId);
        Integer total = clickHouseJdbcTemplate.queryForObject(countSql, Integer.class, countParams.toArray());
        return ResponseEntity.ok(buildPageResponse(results, total, offset, limit));
    }

    @GetMapping("/dashboard")
    public ResponseEntity<Map<String, Object>> getDashboardStats(@RequestParam(defaultValue = "24") Integer hours) {
        try {
            Map<String, Object> stats = new LinkedHashMap<>();
            stats.put("orders", clickHouseJdbcTemplate.queryForMap(ORDER_STATS_SQL, hours));
            stats.put("inventory", clickHouseJdbcTemplate.queryForMap(INVENTORY_STATS_SQL, hours));
            stats.put("userBehavior", clickHouseJdbcTemplate.queryForMap(BEHAVIOR_STATS_SQL, hours));
            stats.put("topViewedProducts", clickHouseJdbcTemplate.queryForList(TOP_VIEWED_SQL, hours));
            stats.put("ordersByHour", clickHouseJdbcTemplate.queryForList(ORDERS_BY_HOUR_SQL, hours));
            stats.put("timeRange", hours + "h");
            stats.put("generatedAt", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            log.error("Failed to query dashboard stats", e);
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        try {
            clickHouseJdbcTemplate.queryForObject("SELECT 1", Integer.class);
            Map<String, Object> health = new LinkedHashMap<>();
            health.put("clickhouse", "UP");
            health.put("timestamp", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            return ResponseEntity.ok(health);
        } catch (Exception e) {
            log.error("ClickHouse health check failed", e);
            Map<String, Object> health = new LinkedHashMap<>();
            health.put("clickhouse", "DOWN");
            health.put("error", e.getMessage());
            health.put("timestamp", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            return ResponseEntity.internalServerError().body(health);
        }
    }

    @GetMapping("/production")
    public ResponseEntity<Map<String, Object>> getProductionAnalytics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        try {
            OlapQueryModels.TimeGranularity granularity = OlapQueryModels.TimeGranularity.DAY;
            List<String> metrics = List.of("output_qty", "qualified_qty", "defect_qty");
            OlapQueryModels.OlapQueryResult result = olapQueryService.rollUp(
                    OlapQueryModels.OlapDomain.PRODUCTION, granularity, null, metrics, startDate, endDate);

            List<Map<String, Object>> dailyData = result.getRows();
            long totalOutput = 0;
            long totalQualified = 0;
            long totalDefects = 0;
            for (Map<String, Object> row : dailyData) {
                totalOutput += ((Number) row.getOrDefault("output_qty", 0)).longValue();
                totalQualified += ((Number) row.getOrDefault("qualified_qty", 0)).longValue();
                totalDefects += ((Number) row.getOrDefault("defect_qty", 0)).longValue();
            }

            Map<String, Object> stats = new LinkedHashMap<>();
            stats.put("dailyData", dailyData);
            stats.put("totalOutput", totalOutput);
            stats.put("totalQualified", totalQualified);
            stats.put("defectCount", totalDefects);
            stats.put("yieldRate", totalOutput > 0 ? (double) totalQualified / totalOutput : 0);
            stats.put("generatedAt", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            log.error("Failed to query production analytics", e);
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    private String buildSelectSql(String columns, String table, List<Object> params,
                                  LocalDateTime start, LocalDateTime end,
                                  String eventType, Long userId, Long productId,
                                  int offset, int limit) {
        StringBuilder sql = new StringBuilder("SELECT ");
        sql.append(columns).append(" FROM ").append(table).append(" WHERE 1=1 ");
        appendTimeFilter(sql, params, start, end);
        appendStringFilter(sql, params, "event_type", eventType);
        appendLongFilter(sql, params, "user_id", userId);
        appendLongFilter(sql, params, "product_id", productId);
        sql.append("ORDER BY event_time DESC LIMIT ? OFFSET ? ");
        params.add(limit);
        params.add(offset);
        return sql.toString();
    }

    private String buildCountSql(String table, List<Object> params,
                                 LocalDateTime start, LocalDateTime end,
                                 String eventType, Long userId, Long productId) {
        StringBuilder sql = new StringBuilder("SELECT COUNT(*) as total FROM ");
        sql.append(table).append(" WHERE 1=1 ");
        appendTimeFilter(sql, params, start, end);
        appendStringFilter(sql, params, "event_type", eventType);
        appendLongFilter(sql, params, "user_id", userId);
        appendLongFilter(sql, params, "product_id", productId);
        return sql.toString();
    }

    private void appendTimeFilter(StringBuilder sql, List<Object> params, LocalDateTime start, LocalDateTime end) {
        if (start != null) {
            sql.append("AND event_time >= ? ");
            params.add(Timestamp.valueOf(start));
        }
        if (end != null) {
            sql.append("AND event_time <= ? ");
            params.add(Timestamp.valueOf(end));
        }
    }

    private void appendStringFilter(StringBuilder sql, List<Object> params, String column, String value) {
        if (value != null && !value.isEmpty()) {
            sql.append("AND ").append(column).append(" = ? ");
            params.add(value);
        }
    }

    private void appendLongFilter(StringBuilder sql, List<Object> params, String column, Long value) {
        if (value != null) {
            sql.append("AND ").append(column).append(" = ? ");
            params.add(value);
        }
    }

    private Map<String, Object> buildPageResponse(List<Map<String, Object>> data, Integer total, Integer offset, Integer limit) {
        Map<String, Object> response = new HashMap<>();
        response.put("data", data);
        response.put("total", total);
        response.put("offset", offset);
        response.put("limit", limit);
        return response;
    }
}
