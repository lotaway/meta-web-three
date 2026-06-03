package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.dto.ApiUsageStatsResponse;
import com.metawebthree.developerportal.entity.ApiUsageStats;
import com.metawebthree.developerportal.repository.ApiUsageStatsRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * API Usage Statistics Service
 * Handles usage tracking, billing, and quota management
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiUsageStatsService {

    private final ApiUsageStatsRepository statsRepository;

    /**
     * Record an API call
     */
    @Transactional
    public void recordApiCall(
        String developerId,
        String keyId,
        String apiEndpoint,
        String httpMethod,
        boolean success,
        long responseTimeMs,
        long dataTransferredBytes,
        long billingAmountCents
    ) {
        // Round to hour for aggregation
        LocalDateTime statTime = LocalDateTime.now().truncatedTo(ChronoUnit.HOURS);
        
        // Find or create stats record
        List<ApiUsageStats> existing = statsRepository.findByDeveloperIdAndApiEndpointAndStatTimeBetween(
            developerId, apiEndpoint, statTime, statTime.plusHours(1)
        );
        
        ApiUsageStats stats;
        if (existing.isEmpty()) {
            stats = new ApiUsageStats();
            stats.setDeveloperId(developerId);
            stats.setKeyId(keyId);
            stats.setApiEndpoint(apiEndpoint);
            stats.setHttpMethod(httpMethod);
            stats.setStatTime(statTime);
            stats.setRequestCount(1L);
            stats.setSuccessCount(success ? 1L : 0L);
            stats.setErrorCount(success ? 0L : 1L);
            stats.setAvgResponseTimeMs((double) responseTimeMs);
            stats.setDataTransferredBytes(dataTransferredBytes);
            stats.setBillingAmountCents(billingAmountCents);
        } else {
            stats = existing.get(0);
            stats.setRequestCount(stats.getRequestCount() + 1);
            if (success) {
                stats.setSuccessCount(stats.getSuccessCount() + 1);
            } else {
                stats.setErrorCount(stats.getErrorCount() + 1);
            }
            // Update average response time
            double prevAvg = stats.getAvgResponseTimeMs() != null ? stats.getAvgResponseTimeMs() : 0;
            long prevCount = stats.getRequestCount() - 1;
            stats.setAvgResponseTimeMs(
                (prevAvg * prevCount + responseTimeMs) / stats.getRequestCount()
            );
            stats.setDataTransferredBytes(stats.getDataTransferredBytes() + dataTransferredBytes);
            stats.setBillingAmountCents(stats.getBillingAmountCents() + billingAmountCents);
        }
        
        statsRepository.save(stats);
    }

    /**
     * Get usage statistics for a developer in time range
     */
    public List<ApiUsageStatsResponse> getDeveloperStats(
        String developerId,
        LocalDateTime startTime,
        LocalDateTime endTime
    ) {
        List<ApiUsageStats> stats = statsRepository.findByDeveloperIdAndStatTimeBetween(
            developerId, startTime, endTime
        );
        
        return stats.stream()
            .map(this::toResponse)
            .collect(Collectors.toList());
    }

    /**
     * Get aggregated usage summary for a developer
     */
    public Map<String, Object> getDeveloperUsageSummary(
        String developerId,
        LocalDateTime startTime,
        LocalDateTime endTime
    ) {
        List<ApiUsageStats> stats = statsRepository.findByDeveloperIdAndStatTimeBetween(
            developerId, startTime, endTime
        );
        
        long totalRequests = stats.stream()
            .mapToLong(ApiUsageStats::getRequestCount)
            .sum();
        
        long totalSuccess = stats.stream()
            .mapToLong(ApiUsageStats::getSuccessCount)
            .sum();
        
        long totalErrors = stats.stream()
            .mapToLong(ApiUsageStats::getErrorCount)
            .sum();
        
        double avgResponseTime = stats.stream()
            .filter(s -> s.getAvgResponseTimeMs() != null)
            .mapToDouble(ApiUsageStats::getAvgResponseTimeMs)
            .average()
            .orElse(0.0);
        
        long totalDataTransferred = stats.stream()
            .mapToLong(ApiUsageStats::getDataTransferredBytes)
            .sum();
        
        long totalBilling = statsRepository.sumBillingAmountByDeveloperAndTimeRange(
            developerId, startTime, endTime
        );
        
        Map<String, Object> summary = new HashMap<>();
        summary.put("developerId", developerId);
        summary.put("startTime", startTime);
        summary.put("endTime", endTime);
        summary.put("totalRequests", totalRequests);
        summary.put("successCount", totalSuccess);
        summary.put("errorCount", totalErrors);
        summary.put("errorRate", totalRequests > 0 ? (double) totalErrors / totalRequests : 0.0);
        summary.put("avgResponseTimeMs", avgResponseTime);
        summary.put("dataTransferredBytes", totalDataTransferred);
        summary.put("billingAmountCents", totalBilling);
        summary.put("billingAmountFormatted", formatCurrency(totalBilling));
        
        return summary;
    }

    /**
     * Get top API endpoints for a developer
     */
    public List<Map<String, Object>> getTopEndpoints(
        String developerId,
        LocalDateTime startTime,
        LocalDateTime endTime,
        int limit
    ) {
        List<Object[]> results = statsRepository.findTopEndpointsByDeveloper(
            developerId, startTime, endTime
        );
        
        return results.stream()
            .limit(limit)
            .map(row -> {
                Map<String, Object> map = new HashMap<>();
                map.put("apiEndpoint", row[0]);
                map.put("requestCount", row[1]);
                return map;
            })
            .collect(Collectors.toList());
    }

    /**
     * Check if developer has exceeded quota
     */
    public boolean hasExceededQuota(String developerId, int dailyQuota, int monthlyQuota) {
        // Check daily quota
        LocalDateTime dayStart = LocalDateTime.now().truncatedTo(ChronoUnit.DAYS);
        Long dailyRequests = statsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, dayStart, dayStart.plusDays(1)
        );
        
        if (dailyRequests != null && dailyRequests >= dailyQuota) {
            log.warn("Developer {} exceeded daily quota: {}/{}", developerId, dailyRequests, dailyQuota);
            return true;
        }
        
        // Check monthly quota
        LocalDateTime monthStart = LocalDateTime.now().withDayOfMonth(1).truncatedTo(ChronoUnit.DAYS);
        Long monthlyRequests = statsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, monthStart, monthStart.plusMonths(1)
        );
        
        if (monthlyRequests != null && monthlyRequests >= monthlyQuota) {
            log.warn("Developer {} exceeded monthly quota: {}/{}", developerId, monthlyRequests, monthlyQuota);
            return true;
        }
        
        return false;
    }

    /**
     * Get current usage counts
     */
    public Map<String, Long> getCurrentUsage(String developerId) {
        LocalDateTime dayStart = LocalDateTime.now().truncatedTo(ChronoUnit.DAYS);
        LocalDateTime monthStart = LocalDateTime.now().withDayOfMonth(1).truncatedTo(ChronoUnit.DAYS);
        
        Long dailyRequests = statsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, dayStart, dayStart.plusDays(1)
        );
        
        Long monthlyRequests = statsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, monthStart, monthStart.plusMonths(1)
        );
        
        Map<String, Long> usage = new HashMap<>();
        usage.put("dailyRequests", dailyRequests != null ? dailyRequests : 0L);
        usage.put("monthlyRequests", monthlyRequests != null ? monthlyRequests : 0L);
        
        return usage;
    }

    /**
     * Convert entity to response DTO
     */
    private ApiUsageStatsResponse toResponse(ApiUsageStats stats) {
        ApiUsageStatsResponse response = new ApiUsageStatsResponse();
        response.setDeveloperId(stats.getDeveloperId());
        response.setApiEndpoint(stats.getApiEndpoint());
        response.setHttpMethod(stats.getHttpMethod());
        response.setTotalRequests(stats.getRequestCount());
        response.setSuccessCount(stats.getSuccessCount());
        response.setErrorCount(stats.getErrorCount());
        response.setAvgResponseTimeMs(stats.getAvgResponseTimeMs());
        response.setDataTransferredBytes(stats.getDataTransferredBytes());
        response.setBillingAmountCents(stats.getBillingAmountCents());
        response.setErrorRate(
            stats.getRequestCount() > 0 
                ? (double) stats.getErrorCount() / stats.getRequestCount() 
                : 0.0
        );
        return response;
    }

    /**
     * Format cents to currency string
     */
    private String formatCurrency(long cents) {
        return String.format("$%.2f", cents / 100.0);
    }
}
