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

@Slf4j
@Service
@RequiredArgsConstructor
public class ApiBillingService {

    private static final int BASIC_CENTS_PER_REQUEST = 1;
    private static final int PROFESSIONAL_CENTS_PER_REQUEST = 2;
    private static final int ENTERPRISE_CENTS_PER_REQUEST = 5;
    private static final double DATA_COST_CENTS_PER_MB = 1.0;
    private static final int SLOW_REQUEST_MS_THRESHOLD = 1000;
    private static final int SLOW_REQUEST_PENALTY_CENTS = 1;

    private final ApiUsageStatsRepository usageStatsRepository;
    private final ApiDeveloperRepository developerRepository;

    public boolean hasExceededQuota(String developerId, int dailyQuota, int monthlyQuota) {
        LocalDateTime now = LocalDateTime.now();

        LocalDateTime startOfDay = now.toLocalDate().atStartOfDay();
        Long dailyUsage = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyUsage = dailyUsage != null ? dailyUsage : 0;

        if (dailyUsage >= dailyQuota) {
            log.warn("Developer {} exceeded daily quota: {}/{}", developerId, dailyUsage, dailyQuota);
            return true;
        }

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

    public long calculateBillingAmount(String developerId, String apiEndpoint, 
                                       int responseTimeMs, int dataTransferredBytes) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        BillingPlan plan = developer.getBillingPlan();

        if (plan == BillingPlan.FREE) {
            return 0L;
        }

        long amount = 0L;

        switch (plan) {
            case BASIC:
                amount += BASIC_CENTS_PER_REQUEST;
                break;
            case PROFESSIONAL:
                amount += PROFESSIONAL_CENTS_PER_REQUEST;
                break;
            case ENTERPRISE:
                amount += ENTERPRISE_CENTS_PER_REQUEST;
                break;
        }

        double dataMb = dataTransferredBytes / (1024.0 * 1024.0);
        amount += (long) (dataMb * DATA_COST_CENTS_PER_MB);

        if (responseTimeMs > SLOW_REQUEST_MS_THRESHOLD) {
            amount += SLOW_REQUEST_PENALTY_CENTS;
        }

        return amount;
    }

    @Transactional
    public void recordApiUsage(String developerId, String keyId, String apiEndpoint,
                               String httpMethod, boolean success, int responseTimeMs,
                               int dataTransferredBytes) {
        long billingAmount = calculateBillingAmount(developerId, apiEndpoint, 
                                                  responseTimeMs, dataTransferredBytes);

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

    public BillingSummary getBillingSummary(String developerId) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startOfMonth = YearMonth.now().atDay(1).atStartOfDay();
        LocalDateTime startOfDay = now.toLocalDate().atStartOfDay();

        Long dailyRequests = usageStatsRepository.sumRequestCountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyRequests = dailyRequests != null ? dailyRequests : 0;

        Long dailyBilling = usageStatsRepository.sumBillingAmountByDeveloperAndTimeRange(
            developerId, startOfDay, now);
        dailyBilling = dailyBilling != null ? dailyBilling : 0;

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

    public List<ApiUsageStats> getUsageStats(String developerId, LocalDateTime startTime, LocalDateTime endTime) {
        return usageStatsRepository.findByDeveloperIdAndStatTimeBetweenOrderByStatTimeDesc(
            developerId, startTime, endTime);
    }

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
