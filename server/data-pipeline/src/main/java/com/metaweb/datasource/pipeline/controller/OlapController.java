package com.metaweb.datasource.pipeline.controller;

import com.metaweb.datasource.pipeline.service.OlapQueryModels.*;
import com.metaweb.datasource.pipeline.service.OlapQueryService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/olap")
public class OlapController {

    @Autowired
    private OlapQueryService olapQueryService;

    @GetMapping("/metadata")
    public ResponseEntity<Map<String, Object>> getMetadata() {
        return ResponseEntity.ok(olapQueryService.getMetadata());
    }

    @PostMapping("/query")
    public ResponseEntity<OlapQueryResult> executeQuery(@RequestBody OlapQueryRequest request) {
        return ResponseEntity.ok(olapQueryService.executeQuery(request));
    }

    @GetMapping("/drill-down")
    public ResponseEntity<DrillDownResult> drillDown(
            @RequestParam OlapDomain domain,
            @RequestParam String parentDimension,
            @RequestParam String parentValue,
            @RequestParam String childDimension,
            @RequestParam(required = false) List<String> metrics,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.drillDown(domain, parentDimension, parentValue,
                childDimension, metrics, startTime, endTime));
    }

    @GetMapping("/roll-up")
    public ResponseEntity<OlapQueryResult> rollUp(
            @RequestParam OlapDomain domain,
            @RequestParam TimeGranularity granularity,
            @RequestParam(required = false) List<String> dimensions,
            @RequestParam(required = false) List<String> metrics,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.rollUp(domain, granularity, dimensions, metrics, startTime, endTime));
    }

    @GetMapping("/slice")
    public ResponseEntity<OlapQueryResult> slice(
            @RequestParam OlapDomain domain,
            @RequestParam String fixedDimension,
            @RequestParam String fixedValue,
            @RequestParam String viewDimension,
            @RequestParam(required = false) List<String> metrics,
            @RequestParam(required = false) TimeGranularity timeGranularity,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.slice(domain, fixedDimension, fixedValue,
                viewDimension, metrics, timeGranularity, startTime, endTime));
    }

    @PostMapping("/dice")
    public ResponseEntity<OlapQueryResult> dice(@RequestBody DiceRequest request) {
        return ResponseEntity.ok(olapQueryService.dice(request.getDomain(), request.getFilters(),
                request.getDimensions(), request.getMetrics(), request.getTimeGranularity(),
                request.getStartTime(), request.getEndTime()));
    }

    @GetMapping("/pivot")
    public ResponseEntity<Map<String, Object>> pivot(
            @RequestParam OlapDomain domain,
            @RequestParam String rowDimension,
            @RequestParam String colDimension,
            @RequestParam(defaultValue = "count") String metric,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.pivot(domain, rowDimension, colDimension, metric, startTime, endTime));
    }

    @GetMapping("/sales-funnel")
    public ResponseEntity<Map<String, Object>> getSalesFunnel(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.getSalesFunnel(startTime, endTime));
    }

    @GetMapping("/cohort-retention")
    public ResponseEntity<List<Map<String, Object>>> getCohortRetention(
            @RequestParam(defaultValue = "6") Integer cohortDays,
            @RequestParam(defaultValue = "3") Integer retentionDays) {
        return ResponseEntity.ok(olapQueryService.getCohortRetention(cohortDays, retentionDays));
    }

    @GetMapping("/top-n")
    public ResponseEntity<List<Map<String, Object>>> getTopN(
            @RequestParam OlapDomain domain,
            @RequestParam String dimension,
            @RequestParam(defaultValue = "count") String metric,
            @RequestParam(defaultValue = "10") Integer n,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return ResponseEntity.ok(olapQueryService.getTopN(domain, dimension, metric, n, startTime, endTime));
    }

    @lombok.Data
    public static class DiceRequest {
        private OlapDomain domain;
        private List<OlapFilter> filters;
        private List<String> dimensions;
        private List<String> metrics;
        private TimeGranularity timeGranularity;
        @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
        private LocalDateTime startTime;
        @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
        private LocalDateTime endTime;
    }
}
