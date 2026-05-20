package com.metawebthree.payment.application;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.metawebthree.common.generated.rpc.RiskScorerService;
import com.metawebthree.common.generated.rpc.UserRiskProfileService;
import com.metawebthree.common.generated.rpc.UserRiskProfile;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileRequest;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileResponse;
import com.metawebthree.common.generated.rpc.ScoreRequest;
import com.metawebthree.common.generated.rpc.ScoreResponse;
import com.metawebthree.common.generated.rpc.Feature;
import com.metawebthree.common.generated.rpc.DeviceRiskTag;
import org.apache.dubbo.config.annotation.DubboReference;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class RiskControlServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;

    @DubboReference(check = false, lazy = true)
    private RiskScorerService riskScorerService;

    @DubboReference(check = false, lazy = true)
    private UserRiskProfileService userRiskProfileService;

    @Value("${payment.risk-control.single-limit.usd:10000}")
    private BigDecimal singleLimitUSD;

    @Value("${payment.rist-control.hourly-order-limit:100}")
    private Integer hourlyOrderLimit;

    @Value("${payment.risk-control.daily-limit.usd:50000}")
    private BigDecimal dailyLimitUSD;

    @Value("${payment.risk-control.slippage.max-percentage:2.0}")
    private BigDecimal maxSlippagePercentage;

    @Value("${payment.risk-control.min-score:600}")
    private int minScore;

    @LogMethod
    public void validateOrder(Long userId, BigDecimal amount, String fiatCurrency) {
        checkRiskScore(userId, amount, fiatCurrency);
        validateSingleLimit(amount, fiatCurrency);
        validateDailyLimit(userId, amount, fiatCurrency);
        validateFrequency(userId);
        validateAbnormalBehavior(userId);
    }

    private void checkRiskScore(Long userId, BigDecimal amount, String fiatCurrency) {
        try {
            GetUserRiskProfileRequest request = GetUserRiskProfileRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetUserRiskProfileResponse response = userRiskProfileService.getUserRiskProfile(request);
            UserRiskProfile userProfile = response.getProfile();

            Map<String, Feature> features = new HashMap<>();

            if (userProfile.getAge() != 0) {
                features.put("age", Feature.newBuilder().setAge(userProfile.getAge()).build());
            }
            if (userProfile.getExternalDebtRatio() != 0.0f) {
                features.put("external_debt_ratio",
                        Feature.newBuilder().setExternalDebtRatio(userProfile.getExternalDebtRatio()).build());
            }
            if (userProfile.getGpsStability() != 0.0f) {
                features.put("gps_stability",
                        Feature.newBuilder().setGpsStability(userProfile.getGpsStability()).build());
            }
            if (userProfile.getDeviceSharedDegree() != 0) {
                features.put("device_shared_degree",
                        Feature.newBuilder().setDeviceSharedDegree(userProfile.getDeviceSharedDegree()).build());
            }
            if (userProfile.getDeviceRiskTag() != DeviceRiskTag.UNKNOWN) {
                features.put("device_risk_tag",
                        Feature.newBuilder().setDeviceRiskTag(userProfile.getDeviceRiskTag()).build());
            }

            features.put("first_order", Feature.newBuilder().setFirstOrder(false).build());

            ScoreRequest scoreRequest = ScoreRequest.newBuilder()
                    .setScene("payment_execution")
                    .putAllFeatures(features)
                    .build();

            ScoreResponse scoreResponse = riskScorerService.score(scoreRequest);
            log.info("Risk score for user {}: score={}, decision={}", userId, scoreResponse.getScore(),
                    scoreResponse.getDecision());

            if (scoreResponse.getScore() < minScore) {
                throw new RuntimeException("Risk score too low: " + scoreResponse.getScore() +
                        " (Required: " + minScore + "). Decision: " + scoreResponse.getDecision());
            }
        } catch (Exception e) {
            log.error("Risk score check failed for user {}", userId, e);
            throw new RuntimeException("Risk assessment failed: " + e.getMessage());
        }
    }

    private void validateSingleLimit(BigDecimal amount, String fiatCurrency) {
        BigDecimal usdAmount = convertToUSD(amount, fiatCurrency);

        if (usdAmount.compareTo(singleLimitUSD) > 0) {
            throw new RuntimeException("Single transaction limit exceeded. Limit: " +
                    singleLimitUSD + " USD, Amount: " + usdAmount + " USD");
        }
    }

    private void validateDailyLimit(Long userId, BigDecimal amount, String fiatCurrency) {
        Timestamp startOfDay = Timestamp
                .valueOf(LocalDateTime.now().withHour(0).withMinute(0).withSecond(0).withNano(0));

        BigDecimal dailyTotal = exchangeOrderRepository.getTotalCompletedAmountByUserIdAndDateRange(userId, startOfDay);
        if (dailyTotal == null) {
            dailyTotal = BigDecimal.ZERO;
        }

        BigDecimal usdAmount = convertToUSD(amount, fiatCurrency);
        BigDecimal totalUSD = dailyTotal.add(usdAmount);

        if (totalUSD.compareTo(dailyLimitUSD) > 0) {
            throw new RuntimeException("Daily limit exceeded. Daily total: " +
                    totalUSD + " USD, Limit: " + dailyLimitUSD + " USD");
        }
    }

    private void validateFrequency(Long userId) {
        Timestamp oneHourAgo = Timestamp.valueOf(LocalDateTime.now().minusHours(1));
        Long hourlyCount = exchangeOrderRepository.getCompletedOrderCountByUserIdAndDateRange(userId, oneHourAgo);
        if (hourlyCount > hourlyOrderLimit) {
            StringBuilder sb = new StringBuilder();
            sb.append("Transaction frequency too high. Hourly limit: ")
                    .append(hourlyOrderLimit).append(", Current: ")
                    .append(hourlyCount);
            throw new RuntimeException(sb.toString());
        }
    }

    private void validateAbnormalBehavior(Long userId) {
        List<ExchangeOrder> failedOrders = exchangeOrderRepository.findByUserIdAndStatus(userId, "FAILED");
        if (failedOrders.size() > 5) {
            throw new RuntimeException("Too many failed orders. Failed count: " + failedOrders.size());
        }
    }

    public void validateSlippage(BigDecimal expectedRate, BigDecimal actualRate) {
        BigDecimal slippage = actualRate.subtract(expectedRate).abs()
                .divide(expectedRate, 4, java.math.RoundingMode.HALF_UP)
                .multiply(new BigDecimal("100"));

        if (slippage.compareTo(maxSlippagePercentage) > 0) {
            throw new RuntimeException("Slippage too high. Expected: " + expectedRate +
                    ", Actual: " + actualRate + ", Slippage: " + slippage + "%");
        }
    }

    public void validateWalletAddress(String walletAddress) {
        if (isBlacklistedAddress(walletAddress)) {
            throw new RuntimeException("Wallet address is blacklisted");
        }
        if (!isValidAddressFormat(walletAddress)) {
            throw new RuntimeException("Invalid wallet address format");
        }
    }

    private boolean isBlacklistedAddress(String walletAddress) {
        return false;
    }

    private boolean isValidAddressFormat(String walletAddress) {
        if (walletAddress.matches("^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")) {
            return true;
        }
        if (walletAddress.matches("^0x[a-fA-F0-9]{40}$")) {
            return true;
        }
        return false;
    }

    private BigDecimal convertToUSD(BigDecimal amount, String currency) {
        return switch (currency) {
            case "USD" -> amount;
            case "CNY" -> amount.divide(new BigDecimal("7.0"), 2, java.math.RoundingMode.HALF_UP);
            case "EUR" -> amount.multiply(new BigDecimal("1.1"));
            default -> amount;
        };
    }
}
