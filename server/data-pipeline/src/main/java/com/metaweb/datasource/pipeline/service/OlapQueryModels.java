package com.metaweb.datasource.pipeline.service;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class OlapQueryModels {

    public enum OlapDomain {
        ORDER, INVENTORY, USER_BEHAVIOR, PRODUCTION
    }

    public enum TimeGranularity {
        HOUR("toHour(event_time)", "hour"),
        DAY("toDate(event_time)", "date"),
        WEEK("toMonday(event_time)", "week_start"),
        MONTH("toYYYYMM(event_time)", "month"),
        QUARTER("toYYYYMM(event_time)", "month");

        public final String sqlExpr;
        public final String alias;

        TimeGranularity(String sqlExpr, String alias) {
            this.sqlExpr = sqlExpr;
            this.alias = alias;
        }
    }

    @Data
    @Builder
    public static class OlapQueryRequest {
        private OlapDomain domain;
        private List<String> dimensions;
        private List<String> metrics;
        private List<OlapFilter> filters;
        private TimeGranularity timeGranularity;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Integer limit;
        private String orderBy;
        private Boolean ascending;
    }

    @Data
    @Builder
    public static class OlapFilter {
        private String field;
        private String operator;
        private Object value;
        private Object value2;
    }

    @Data
    @Builder
    public static class OlapQueryResult {
        private List<String> columns;
        private List<Map<String, Object>> rows;
        private Map<String, Object> totals;
        private String sql;
        private long queryTimeMs;
    }

    @Data
    @Builder
    public static class DrillDownResult {
        private String dimension;
        private Object dimensionValue;
        private List<Map<String, Object>> children;
        private Map<String, Object> aggregates;
    }
}
