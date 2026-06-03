package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.dto.ApiUsageStatsResponse;
import com.metawebthree.developerportal.service.ApiUsageStatsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * API Usage Statistics Controller
 * Handles usage tracking and billing queries
 */
@Tag(name = "API Usage Statistics", description = "API usage tracking and billing")
@RestController
@RequestMapping("/developer/usage")
@RequiredArgsConstructor
public class ApiUsageStatsController {

    private final ApiUsageStatsService statsService;

    @Operation(summary = "Get usage statistics", description = "Get detailed usage statistics for a developer in a time range")
    @GetMapping("/{developerId}")
    public ResponseEntity<List<ApiUsageStatsResponse>> getDeveloperStats(
        @PathVariable String developerId,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime
    ) {
        List<ApiUsageStatsResponse> response = statsService.getDeveloperStats(developerId, startTime, endTime);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get usage summary", description = "Get aggregated usage summary for a developer")
    @GetMapping("/{developerId}/summary")
    public ResponseEntity<Map<String, Object>> getDeveloperUsageSummary(
        @PathVariable String developerId,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime
    ) {
        Map<String, Object> response = statsService.getDeveloperUsageSummary(developerId, startTime, endTime);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get top endpoints", description = "Get most frequently used API endpoints for a developer")
    @GetMapping("/{developerId}/top-endpoints")
    public ResponseEntity<List<Map<String, Object>>> getTopEndpoints(
        @PathVariable String developerId,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
        @RequestParam(defaultValue = "10") int limit
    ) {
        List<Map<String, Object>> response = statsService.getTopEndpoints(developerId, startTime, endTime, limit);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get current usage", description = "Get current daily and monthly usage counts")
    @GetMapping("/{developerId}/current")
    public ResponseEntity<Map<String, Long>> getCurrentUsage(@PathVariable String developerId) {
        Map<String, Long> response = statsService.getCurrentUsage(developerId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Check quota status", description = "Check if developer has exceeded quota")
    @GetMapping("/{developerId}/quota-check")
    public ResponseEntity<Map<String, Object>> checkQuota(
        @PathVariable String developerId,
        @RequestParam int dailyQuota,
        @RequestParam int monthlyQuota
    ) {
        boolean exceeded = statsService.hasExceededQuota(developerId, dailyQuota, monthlyQuota);
        Map<String, Long> usage = statsService.getCurrentUsage(developerId);
        
        Map<String, Object> response = Map.of(
            "developerId", developerId,
            "dailyQuota", dailyQuota,
            "monthlyQuota", monthlyQuota,
            "dailyUsage", usage.get("dailyRequests"),
            "monthlyUsage", usage.get("monthlyRequests"),
            "quotaExceeded", exceeded
        );
        
        return ResponseEntity.ok(response);
    }
}
