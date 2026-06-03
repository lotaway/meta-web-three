package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.ApiDeveloper.BillingPlan;
import com.metawebthree.developerportal.entity.ApiUsageStats;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import com.metawebthree.developerportal.repository.ApiUsageStatsRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.YearMonth;
import java.util.List;

/**
 * API Billing Service
 * Handles billing calculation, quota management, and usage tracking
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiBillingService {

    private final ApiUsageStatsRepository usageStatsRepository;
    private final ApiDeveloperRepository developerRepository;

    /**
     * Check if developer has exceeded quota
     * @return true if quota exceeded
     */
    public boolean hasExceededQuota(String developerId, int dailyQuota, int monthlyQuota) {
        LocalDateTime now = LocalDateTime.now();
        
        // Check daily quota
        LocalDateTime startOfDay = now.toLocalDate().atStartOfDay();
        Long dailyUsage = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyUsage = dailyUsage != null ? dailyUsage : 0;
        
        if (dailyUsage >= dailyQuota) {
            log.warn("Developer {} exceeded daily quota: {}/{}", developerId, dailyUsage, dailyQuota);
            return true;
        }
        
        // Check monthly quota
        LocalDateTime startOfMonth = YearMonth.now().atDay(1).atStartOfDay();
        Long monthlyUsage = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfMonth, now);
        monthlyUsage = monthlyUsage != null ? monthlyUsage : 0;
        
        if (monthlyUsage >= monthlyQuota) {
            log.warn("Developer {} exceeded monthly quota: {}/{}", developerId, monthlyUsage, monthlyQuota);
            return true;
        }
        
        return false;
    }

    /**
     * Calculate billing amount for an API call
     * @return billing amount in cents
     */
    public long calculateBillingAmount(String developerId, String apiEndpoint, 
                                      int responseTimeMs, int dataTransferredBytes) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));
        
        BillingPlan plan = developer.getBillingPlan();
        
        // Free tier: no billing
        if (plan == BillingPlan.FREE) {
            return 0L;
        }
        
        // Paid tiers: calculate based on usage
        long amount = 0L;
        
        // Base cost per request (in cents)
        switch (plan) {
            case BASIC:
                amount += 1; // $0.01 per request
                break;
            case PROFESSIONAL:
                amount += 2; // $0.02 per request
                break;
            case ENTERPRISE:
                amount += 5; // $0.05 per request
                break;
        }
        
        // Additional cost for data transfer ($0.01 per MB)
        double dataMb = dataTransferredBytes / (1024.0 * 1024.0);
        amount += (long) (dataMb * 1.0);
        
        // Additional cost for slow responses (performance penalty)
        if (responseTimeMs > 1000) {
            amount += 1; // $0.01 penalty for slow responses
        }
        
        return amount;
    }

    /**
     * Record API usage with billing
     */
    @Transactional
    public void recordApiUsage(String developerId, String keyId, String apiEndpoint,
                              String httpMethod, boolean success, int responseTimeMs,
                              int dataTransferredBytes) {
        // Calculate billing
        long billingAmount = calculateBillingAmount(developerId, apiEndpoint, 
                                                  responseTimeMs, dataTransferredBytes);
        
        // Create usage stats record
        ApiUsageStats stats = new ApiUsageStats();
        stats.setDeveloperId(developerId);
        stats.setKeyId(keyId);
        stats.setApiEndpoint(apiEndpoint);
        stats.setHttpMethod(httpMethod);
        stats.setStatTime(LocalDateTime.now());
        stats.setRequestCount(1L);
        stats.setSuccessCount(success ? 1L : 0L);
        stats.setErrorCount(success ? 0L : 1L);
        stats.setAvgResponseTimeMs((double) responseTimeMs);
        stats.setDataTransferredBytes((long) dataTransferredBytes);
        stats.setBillingAmountCents(billingAmount);
        
        usageStatsRepository.save(stats);
        
        // Deduct from balance (if prepaid)
        if (billingAmount > 0) {
            ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
                .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));
            
            long newBalance = developer.getBalance() - billingAmount;
            if (newBalance < 0) {
                log.warn("Developer {} has insufficient balance: {} - {} = {}", 
                    developerId, developer.getBalance(), billingAmount, newBalance);
                throw new IllegalStateException("Insufficient balance");
            }
            
            developer.setBalance(newBalance);
            developerRepository.save(developer);
        }
        
        log.debug("API usage recorded: developer={}, endpoint={}, billing={} cents", 
            developerId, apiEndpoint, billingAmount);
    }

    /**
     * Get developer's current billing period usage
     */
    public BillingSummary getBillingSummary(String developerId) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));
        
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startOfMonth = YearMonth.now().atDay(1).atStartOfDay();
        LocalDateTime startOfDay = now.toLocalDate().atStartOfDay();
        
        // Daily usage
        Long dailyRequests = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyRequests = dailyRequests != null ? dailyRequests : 0;
        
        Long dailyBilling = usageStatsRepository.sumBillingAmountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyBilling = dailyBilling != null ? dailyBilling : 0;
        
        // Monthly usage
        Long monthlyRequests = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfMonth, now);
        monthlyRequests = monthlyRequests != null ? monthlyRequests : 0;
        
        Long monthlyBilling = usageStatsRepository.sumBillingAmountByDeveloperAndTimeRange(
            developerId, startOfMonth, now);
        monthlyBilling = monthlyBilling != null ? monthlyBilling : 0;
        
        return new BillingSummary(
            developerId,
            dailyRequests,
            dailyBilling,
            developer.getDailyQuota(),
            monthlyRequests,
            monthlyBilling,
            developer.getMonthlyQuota(),
            developer.getBalance()
        );
    }

    /**
     * Get usage statistics for a time range
     */
    public List<ApiUsageStats> getUsageStats(String developerId, LocalDateTime startTime, LocalDateTime endTime) {
        return usageStatsRepository.findByDeveloperIdAndStatTimeBetweenOrderByStatTimeDesc(
            developerId, startTime, endTime);
    }

    /**
     * Billing summary DTO
     */
    public static record BillingSummary(
        String developerId,
        long dailyRequests,
        long dailyBillingCents,
        int dailyQuota,
        long monthlyRequests,
        long monthlyBillingCents,
        int monthlyQuota,
        long balanceCents
    ) {}
}
