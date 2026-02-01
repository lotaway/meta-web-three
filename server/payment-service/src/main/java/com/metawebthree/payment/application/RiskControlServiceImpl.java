package com.metawebthree.payment.application;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 风控服务
 * 
 * TODO: 需要添加 @/risk-scorer 程序进行处理，其次如需接入第三方风控服务（如Chainalysis、Elliptic等），请在
 * validateOrder、validateWalletAddress 等方法中
 * 调用外部API进行合规校验、地址风险识别、异常行为检测等。
 * 推荐将第三方API调用、风控规则配置等逻辑封装为独立方法或类，便于后续维护和切换。
 *
 * 示例：
 * 1. 在 validateOrder 中调用外部风控API
 * 2. 在 validateWalletAddress 中集成地址黑名单/合规检查
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class RiskControlServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;

    @Value("${payment.risk-control.single-limit.usd:10000}")
    private BigDecimal singleLimitUSD;

    @Value("${payment.rist-control.hourly-order-limit:100}")
    private Integer hourlyOrderLimit;

    @Value("${payment.risk-control.daily-limit.usd:50000}")
    private BigDecimal dailyLimitUSD;

    @Value("${payment.risk-control.slippage.max-percentage:2.0}")
    private BigDecimal maxSlippagePercentage;

    /**
     * @TODO Add external risk control service integration.
     */
    @LogMethod
    public void validateOrder(Long userId, BigDecimal amount, String fiatCurrency) {
        validateSingleLimit(amount, fiatCurrency);
        validateDailyLimit(userId, amount, fiatCurrency);
        validateFrequency(userId);
        validateAbnormalBehavior(userId);
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
        // @TODO check for abnormal order patterns, such as order amount, time intervals
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

    private BigDecimal convertToUSD(BigDecimal amount, String currency) {
        return switch (currency) {
            case "USD" -> amount;
            case "CNY" -> amount.divide(new BigDecimal("7.0"), 2, java.math.RoundingMode.HALF_UP);
            case "EUR" -> amount.multiply(new BigDecimal("1.1"));
            default -> amount;
        };
    }

    /**
     * @TODO Integration with third-party address risk identification service
     */
    public void validateWalletAddress(String walletAddress) {
        if (isBlacklistedAddress(walletAddress)) {
            throw new RuntimeException("Wallet address is blacklisted");
        }
        if (!isValidAddressFormat(walletAddress)) {
            throw new RuntimeException("Invalid wallet address format");
        }
    }

    /**
     * @TODO check black list of wallet address
     */
    private boolean isBlacklistedAddress(String walletAddress) {
        return false;
    }

    private boolean isValidAddressFormat(String walletAddress) {
        // BTC address
        if (walletAddress.matches("^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")) {
            return true;
        }
        // ETH address
        if (walletAddress.matches("^0x[a-fA-F0-9]{40}$")) {
            return true;
        }
        return false;
    }
}
