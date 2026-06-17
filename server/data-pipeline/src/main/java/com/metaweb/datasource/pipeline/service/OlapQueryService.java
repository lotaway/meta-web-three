package com.metaweb.datasource.pipeline.service;

import com.metaweb.datasource.pipeline.service.OlapQueryModels.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Slf4j
@Service
public class OlapQueryService {

    @Autowired
    @org.springframework.beans.factory.annotation.Qualifier("clickHouseJdbcTemplate")
    private JdbcTemplate clickHouseJdbcTemplate;

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final int DEFAULT_LIMIT = 100;

    private static final Map<OlapDomain, String> DOMAIN_TABLE = Map.of(
            OlapDomain.ORDER, "meta_web_analytics.order_analytics",
            OlapDomain.INVENTORY, "meta_web_analytics.inventory_analytics",
            OlapDomain.USER_BEHAVIOR, "meta_web_analytics.user_behavior_analytics"
    );

    private static final Map<OlapDomain, List<String>> DOMAIN_DIMENSIONS = Map.of(
            OlapDomain.ORDER, List.of("event_type", "status", "payment_method", "year_month", "day_of_week", "hour_of_day", "user_id", "merchant_id"),
            OlapDomain.INVENTORY, List.of("event_type", "warehouse_id", "product_id", "operator"),
            OlapDomain.USER_BEHAVIOR, List.of("event_type", "device_type", "browser_family", "os", "category")
    );

    private static final Map<OlapDomain, List<String>> DOMAIN_METRICS = Map.of(
            OlapDomain.ORDER, List.of("count", "total_amount_sum", "total_amount_avg", "unique_users", "unique_orders"),
            OlapDomain.INVENTORY, List.of("count", "quantity_sum", "available_qty_avg", "unique_products"),
            OlapDomain.USER_BEHAVIOR, List.of("count", "unique_users", "unique_sessions", "duration_avg", "duration_sum")
    );

    public OlapQueryResult executeQuery(OlapQueryRequest request) {
        long start = System.currentTimeMillis();
        List<Object> params = new ArrayList<>();
        String selectClause = buildSelectClause(request);
        String fromWhereClause = buildFromWhereClause(request, params);
        String groupByClause = buildGroupByClause(request);
        String orderByClause = buildOrderByClause(request);
        String limitClause = buildLimitClause(request, params);
        String sql = selectClause + fromWhereClause + groupByClause + orderByClause + limitClause;
        log.info("OLAP query: {}", sql);
        List<Map<String, Object>> rows = clickHouseJdbcTemplate.queryForList(sql, params.toArray());
        long elapsed = System.currentTimeMillis() - start;
        List<String> columns = buildColumnList(request);
        return OlapQueryResult.builder().columns(columns).rows(rows).sql(sql).queryTimeMs(elapsed).build();
    }

    public DrillDownResult drillDown(OlapDomain domain, String parentDimension, Object parentValue,
                                      String childDimension, List<String> metrics,
                                      LocalDateTime startTime, LocalDateTime endTime) {
        List<Object> params = new ArrayList<>();
        String sql = buildDrillDownSql(domain, parentDimension, parentValue, childDimension, metrics, params, startTime, endTime);
        List<Map<String, Object>> children = clickHouseJdbcTemplate.queryForList(sql, params.toArray());
        Map<String, Object> aggregates = computeAggregates(domain, metrics, parentDimension, parentValue, startTime, endTime);
        return DrillDownResult.builder()
                .dimension(parentDimension).dimensionValue(parentValue)
                .children(children).aggregates(aggregates)
                .build();
    }

    public OlapQueryResult rollUp(OlapDomain domain, TimeGranularity granularity,
                                   List<String> dimensions, List<String> metrics,
                                   LocalDateTime startTime, LocalDateTime endTime) {
        OlapQueryRequest request = OlapQueryRequest.builder()
                .domain(domain).timeGranularity(granularity).dimensions(dimensions).metrics(metrics)
                .startTime(startTime).endTime(endTime)
                .build();
        return executeQuery(request);
    }

    public OlapQueryResult slice(OlapDomain domain, String fixedDimension, Object fixedValue,
                                  String viewDimension, List<String> metrics,
                                  TimeGranularity timeGranularity,
                                  LocalDateTime startTime, LocalDateTime endTime) {
        OlapFilter sliceFilter = OlapFilter.builder()
                .field(fixedDimension).operator("EQ").value(fixedValue)
                .build();
        OlapQueryRequest request = OlapQueryRequest.builder()
                .domain(domain).timeGranularity(timeGranularity)
                .dimensions(List.of(viewDimension)).metrics(metrics)
                .filters(List.of(sliceFilter))
                .startTime(startTime).endTime(endTime)
                .build();
        return executeQuery(request);
    }

    public OlapQueryResult dice(OlapDomain domain, List<OlapFilter> filters,
                                 List<String> dimensions, List<String> metrics,
                                 TimeGranularity timeGranularity,
                                 LocalDateTime startTime, LocalDateTime endTime) {
        OlapQueryRequest request = OlapQueryRequest.builder()
                .domain(domain).timeGranularity(timeGranularity).dimensions(dimensions).metrics(metrics)
                .filters(filters).startTime(startTime).endTime(endTime)
                .build();
        return executeQuery(request);
    }

    public Map<String, Object> pivot(OlapDomain domain, String rowDimension, String colDimension,
                                      String metric, LocalDateTime startTime, LocalDateTime endTime) {
        List<Object> colParams = new ArrayList<>();
        String colValuesSql = buildColValuesSql(domain, colDimension, colParams, startTime, endTime);
        List<Map<String, Object>> colValueRows = clickHouseJdbcTemplate.queryForList(colValuesSql, colParams.toArray());
        List<String> colValues = extractColValues(colValueRows, colDimension);
        List<Object> pivotParams = new ArrayList<>();
        String pivotSql = buildPivotSql(domain, rowDimension, colDimension, colValues, metric, pivotParams, startTime, endTime);
        List<Map<String, Object>> rows = clickHouseJdbcTemplate.queryForList(pivotSql, pivotParams.toArray());
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("rowDimension", rowDimension);
        result.put("colDimension", colDimension);
        result.put("colValues", colValues);
        result.put("metric", metric);
        result.put("data", rows);
        return result;
    }

    public Map<String, Object> getSalesFunnel(LocalDateTime startTime, LocalDateTime endTime) {
        List<Object> params = new ArrayList<>();
        String where = buildWhereClause(params, startTime, endTime);
        Map<String, Object> funnel = new LinkedHashMap<>();
        funnel.put("views", clickHouseJdbcTemplate.queryForMap(
                "SELECT COUNT(*) as views, COUNT(DISTINCT user_id) as unique_viewers FROM meta_web_analytics.user_behavior_analytics " + where + " AND event_type = 'PRODUCT_VIEW'", params.toArray()));
        funnel.put("addToCart", clickHouseJdbcTemplate.queryForMap(
                "SELECT COUNT(*) as add_to_cart, COUNT(DISTINCT user_id) as unique_cart_users FROM meta_web_analytics.user_behavior_analytics " + where + " AND event_type = 'ADD_TO_CART'", params.toArray()));
        List<Object> orderParams = new ArrayList<>(params);
        funnel.put("orders", clickHouseJdbcTemplate.queryForMap(
                "SELECT COUNT(*) as orders, COUNT(DISTINCT user_id) as unique_buyers, SUM(total_amount) as total_amount FROM meta_web_analytics.order_analytics " + where + " AND event_type = 'CREATE'", orderParams.toArray()));
        funnel.put("payments", clickHouseJdbcTemplate.queryForMap(
                "SELECT COUNT(*) as payments, SUM(total_amount) as paid_amount FROM meta_web_analytics.order_analytics " + where + " AND status = 'PAID'", orderParams.toArray()));
        return funnel;
    }

    public List<Map<String, Object>> getCohortRetention(int cohortDays, int retentionDays) {
        String sql = """
            SELECT first_order_month AS cohort_month, COUNT(DISTINCT user_id) AS cohort_size,
                   sumIf(1, months_since_first = 1) AS m1,
                   sumIf(1, months_since_first = 2) AS m2,
                   sumIf(1, months_since_first = 3) AS m3
            FROM (SELECT user_id, toYYYYMM(first_value(event_time) OVER (PARTITION BY user_id ORDER BY event_time)) AS first_order_month,
                         toYYYYMM(event_time) - toYYYYMM(first_value(event_time) OVER (PARTITION BY user_id ORDER BY event_time)) AS months_since_first
                  FROM meta_web_analytics.order_analytics WHERE event_type = 'CREATE')
            GROUP BY first_order_month ORDER BY first_order_month LIMIT ?
            """;
        return clickHouseJdbcTemplate.queryForList(sql, cohortDays);
    }

    public List<Map<String, Object>> getTopN(OlapDomain domain, String dimension, String metric,
                                              int n, LocalDateTime startTime, LocalDateTime endTime) {
        String table = DOMAIN_TABLE.get(domain);
        String metricExpr = buildSingleMetricExpression(domain, metric);
        List<Object> params = new ArrayList<>();
        StringBuilder sql = new StringBuilder("SELECT ");
        sql.append(dimension).append(", ").append(metricExpr).append(" AS ").append(metric);
        sql.append(" FROM ").append(table).append(" WHERE 1=1 ");
        appendTimeRange(sql, params, startTime, endTime);
        sql.append("GROUP BY ").append(dimension).append(" ");
        sql.append("ORDER BY ").append(metric).append(" DESC LIMIT ? ");
        params.add(n);
        return clickHouseJdbcTemplate.queryForList(sql.toString(), params.toArray());
    }

    public Map<String, Object> getMetadata() {
        Map<String, Object> meta = new LinkedHashMap<>();
        for (OlapDomain domain : OlapDomain.values()) {
            Map<String, Object> domainMeta = new LinkedHashMap<>();
            domainMeta.put("table", DOMAIN_TABLE.get(domain));
            domainMeta.put("dimensions", DOMAIN_DIMENSIONS.get(domain));
            domainMeta.put("metrics", DOMAIN_METRICS.get(domain));
            meta.put(domain.name(), domainMeta);
        }
        meta.put("timeGranularities", Arrays.stream(TimeGranularity.values()).map(Enum::name).toList());
        return meta;
    }

    private String buildSelectClause(OlapQueryRequest request) {
        StringBuilder sql = new StringBuilder("SELECT ");
        if (request.getTimeGranularity() != null) {
            sql.append(request.getTimeGranularity().sqlExpr).append(" AS ").append(request.getTimeGranularity().alias).append(", ");
        }
        if (request.getDimensions() != null) {
            for (String dim : request.getDimensions()) {
                sql.append(dim).append(", ");
            }
        }
        sql.append(String.join(", ", buildMetricExpressions(request.getDomain(), request.getMetrics())));
        return sql.toString();
    }

    private String buildFromWhereClause(OlapQueryRequest request, List<Object> params) {
        StringBuilder sql = new StringBuilder(" FROM ").append(DOMAIN_TABLE.get(request.getDomain())).append(" WHERE 1=1 ");
        appendTimeRange(sql, params, request.getStartTime(), request.getEndTime());
        if (request.getFilters() != null) {
            for (OlapFilter filter : request.getFilters()) {
                appendFilter(sql, params, filter);
            }
        }
        return sql.toString();
    }

    private String buildGroupByClause(OlapQueryRequest request) {
        List<String> groupByCols = new ArrayList<>();
        if (request.getTimeGranularity() != null) {
            groupByCols.add(request.getTimeGranularity().alias);
        }
        if (request.getDimensions() != null) {
            groupByCols.addAll(request.getDimensions());
        }
        if (groupByCols.isEmpty()) {
            return "";
        }
        return " GROUP BY " + String.join(", ", groupByCols) + " ";
    }

    private String buildOrderByClause(OlapQueryRequest request) {
        if (request.getOrderBy() != null) {
            return " ORDER BY " + request.getOrderBy() + (Boolean.TRUE.equals(request.getAscending()) ? " ASC " : " DESC ");
        }
        List<String> groupByCols = new ArrayList<>();
        if (request.getTimeGranularity() != null) {
            groupByCols.add(request.getTimeGranularity().alias);
        }
        if (request.getDimensions() != null) {
            groupByCols.addAll(request.getDimensions());
        }
        if (!groupByCols.isEmpty()) {
            return " ORDER BY " + groupByCols.get(0) + " ASC ";
        }
        return "";
    }

    private String buildLimitClause(OlapQueryRequest request, List<Object> params) {
        int limit = request.getLimit() != null ? request.getLimit() : DEFAULT_LIMIT;
        params.add(limit);
        return " LIMIT ? ";
    }

    private List<String> buildColumnList(OlapQueryRequest request) {
        List<String> columns = new ArrayList<>();
        if (request.getTimeGranularity() != null) {
            columns.add(request.getTimeGranularity().alias);
        }
        if (request.getDimensions() != null) {
            columns.addAll(request.getDimensions());
        }
        columns.addAll(request.getMetrics() != null ? request.getMetrics() : List.of("count"));
        return columns;
    }

    private String buildDrillDownSql(OlapDomain domain, String parentDimension, Object parentValue,
                                      String childDimension, List<String> metrics,
                                      List<Object> params, LocalDateTime startTime, LocalDateTime endTime) {
        String table = DOMAIN_TABLE.get(domain);
        StringBuilder sql = new StringBuilder("SELECT ");
        sql.append(childDimension).append(", ");
        sql.append(String.join(", ", buildMetricExpressions(domain, metrics)));
        sql.append(" FROM ").append(table).append(" WHERE 1=1 ");
        sql.append("AND ").append(parentDimension).append(" = ? ");
        params.add(parentValue);
        appendTimeRange(sql, params, startTime, endTime);
        sql.append("GROUP BY ").append(childDimension).append(" ORDER BY ").append(childDimension).append(" ASC ");
        return sql.toString();
    }

    private String buildColValuesSql(OlapDomain domain, String colDimension,
                                      List<Object> params, LocalDateTime startTime, LocalDateTime endTime) {
        String table = DOMAIN_TABLE.get(domain);
        StringBuilder sql = new StringBuilder("SELECT DISTINCT ").append(colDimension).append(" FROM ").append(table).append(" WHERE 1=1 ");
        appendTimeRange(sql, params, startTime, endTime);
        sql.append("ORDER BY ").append(colDimension);
        return sql.toString();
    }

    private List<String> extractColValues(List<Map<String, Object>> colValueRows, String colDimension) {
        List<String> colValues = new ArrayList<>();
        for (Map<String, Object> row : colValueRows) {
            colValues.add(String.valueOf(row.get(colDimension)));
        }
        return colValues;
    }

    private String buildPivotSql(OlapDomain domain, String rowDimension, String colDimension,
                                  List<String> colValues, String metric,
                                  List<Object> params, LocalDateTime startTime, LocalDateTime endTime) {
        String table = DOMAIN_TABLE.get(domain);
        String metricExpr = buildSingleMetricExpression(domain, metric);
        StringBuilder sql = new StringBuilder("SELECT ");
        sql.append(rowDimension).append(", ");
        for (int i = 0; i < colValues.size(); i++) {
            if (i > 0) sql.append(", ");
            sql.append("sumIf(").append(metricExpr).append(", ").append(colDimension).append(" = ?) AS `").append(colValues.get(i)).append("`");
            params.add(colValues.get(i));
        }
        sql.append(", sum(").append(metricExpr).append(") AS total");
        sql.append(" FROM ").append(table).append(" WHERE 1=1 ");
        appendTimeRange(sql, params, startTime, endTime);
        sql.append("GROUP BY ").append(rowDimension).append(" ORDER BY ").append(rowDimension);
        return sql.toString();
    }

    private String buildWhereClause(List<Object> params, LocalDateTime startTime, LocalDateTime endTime) {
        StringBuilder where = new StringBuilder("WHERE 1=1 ");
        appendTimeRange(where, params, startTime, endTime);
        return where.toString();
    }

    private void appendTimeRange(StringBuilder sql, List<Object> params, LocalDateTime start, LocalDateTime end) {
        if (start != null) {
            sql.append("AND event_time >= ? ");
            params.add(start.format(FMT));
        }
        if (end != null) {
            sql.append("AND event_time <= ? ");
            params.add(end.format(FMT));
        }
    }

    private void appendFilter(StringBuilder sql, List<Object> params, OlapFilter filter) {
        sql.append("AND ").append(filter.getField()).append(" ");
        switch (filter.getOperator().toUpperCase()) {
            case "EQ" -> { sql.append("= ? "); params.add(filter.getValue()); }
            case "NE" -> { sql.append("!= ? "); params.add(filter.getValue()); }
            case "GT" -> { sql.append("> ? "); params.add(filter.getValue()); }
            case "LT" -> { sql.append("< ? "); params.add(filter.getValue()); }
            case "GTE" -> { sql.append(">= ? "); params.add(filter.getValue()); }
            case "LTE" -> { sql.append("<= ? "); params.add(filter.getValue()); }
            case "IN" -> {
                List<?> values = (List<?>) filter.getValue();
                String placeholders = String.join(",", Collections.nCopies(values.size(), "?"));
                sql.append("IN (").append(placeholders).append(") ");
                params.addAll(values);
            }
            case "LIKE" -> { sql.append("LIKE ? "); params.add("%" + filter.getValue() + "%"); }
            case "BETWEEN" -> { sql.append("BETWEEN ? AND ? "); params.add(filter.getValue()); params.add(filter.getValue2()); }
            default -> { sql.append("= ? "); params.add(filter.getValue()); }
        }
    }

    private List<String> buildMetricExpressions(OlapDomain domain, List<String> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return List.of("count() AS count");
        }
        List<String> exprs = new ArrayList<>();
        for (String metric : metrics) {
            exprs.add(buildSingleMetricExpression(domain, metric) + " AS " + metric);
        }
        return exprs;
    }

    private String buildSingleMetricExpression(OlapDomain domain, String metric) {
        return switch (metric) {
            case "count" -> "count()";
            case "total_amount_sum" -> "sum(total_amount)";
            case "total_amount_avg" -> "avg(total_amount)";
            case "unique_users" -> "uniqExact(user_id)";
            case "unique_orders" -> "uniqExact(order_id)";
            case "unique_sessions" -> "uniqExact(session_id)";
            case "quantity_sum" -> "sum(quantity)";
            case "available_qty_avg" -> "avg(available_qty)";
            case "unique_products" -> "uniqExact(product_id)";
            case "duration_avg" -> "avg(duration)";
            case "duration_sum" -> "sum(duration)";
            default -> "count()";
        };
    }

    private Map<String, Object> computeAggregates(OlapDomain domain, List<String> metrics,
                                                    String parentDim, Object parentValue,
                                                    LocalDateTime startTime, LocalDateTime endTime) {
        String table = DOMAIN_TABLE.get(domain);
        List<Object> params = new ArrayList<>();
        StringBuilder sql = new StringBuilder("SELECT ");
        sql.append(String.join(", ", buildMetricExpressions(domain, metrics)));
        sql.append(" FROM ").append(table).append(" WHERE ").append(parentDim).append(" = ? ");
        params.add(parentValue);
        appendTimeRange(sql, params, startTime, endTime);
        try {
            return clickHouseJdbcTemplate.queryForMap(sql.toString(), params.toArray());
        } catch (Exception e) {
            log.error("Failed to compute aggregates", e);
            return Map.of();
        }
    }
}
