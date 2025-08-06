package com.metawebthree.service;

import com.metawebthree.repository.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 风控服务
 *
 * TODO: 如需接入第三方风控服务（如Chainalysis、Elliptic等），请在 validateOrder、validateWalletAddress 等方法中
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
public class RiskControlService {
    
    private final ExchangeOrderRepository exchangeOrderRepository;
    
    @Value("${payment.risk-control.single-limit.usd:10000}")
    private BigDecimal singleLimitUSD;
    
    @Value("${payment.risk-control.daily-limit.usd:50000}")
    private BigDecimal dailyLimitUSD;
    
    @Value("${payment.risk-control.slippage.max-percentage:2.0}")
    private BigDecimal maxSlippagePercentage;
    
    /**
     * 验证订单风控
     *
     * TODO: 如需扩展风控规则或接入外部风控服务，请在此处实现。
     */
    public void validateOrder(Long userId, BigDecimal amount, String fiatCurrency) {
        // 1. 单笔限额检查
        validateSingleLimit(amount, fiatCurrency);
        
        // 2. 日限额检查
        validateDailyLimit(userId, amount, fiatCurrency);
        
        // 3. 频率检查
        validateFrequency(userId);
        
        // 4. 异常行为检查
        validateAbnormalBehavior(userId);
        
        log.info("Risk control validation passed for user {}: amount={} {}", userId, amount, fiatCurrency);
    }
    
    /**
     * 验证单笔限额
     */
    private void validateSingleLimit(BigDecimal amount, String fiatCurrency) {
        BigDecimal usdAmount = convertToUSD(amount, fiatCurrency);
        
        if (usdAmount.compareTo(singleLimitUSD) > 0) {
            throw new RuntimeException("Single transaction limit exceeded. Limit: " + 
                    singleLimitUSD + " USD, Amount: " + usdAmount + " USD");
        }
    }
    
    /**
     * 验证日限额
     */
    private void validateDailyLimit(Long userId, BigDecimal amount, String fiatCurrency) {
        LocalDateTime startOfDay = LocalDateTime.now().withHour(0).withMinute(0).withSecond(0).withNano(0);
        
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
    
    /**
     * 验证交易频率
     */
    private void validateFrequency(Long userId) {
        LocalDateTime oneHourAgo = LocalDateTime.now().minusHours(1);
        Long hourlyCount = exchangeOrderRepository.getCompletedOrderCountByUserIdAndDateRange(userId, oneHourAgo);
        
        if (hourlyCount > 10) {
            throw new RuntimeException("Transaction frequency too high. Hourly limit: 10, Current: " + hourlyCount);
        }
    }
    
    /**
     * 验证异常行为
     */
    private void validateAbnormalBehavior(Long userId) {
        // 检查是否有失败的订单
        long failedOrders = exchangeOrderRepository.findByUserIdAndStatus(userId, 
                com.metawebthree.entity.ExchangeOrder.OrderStatus.FAILED).size();
        
        if (failedOrders > 5) {
            throw new RuntimeException("Too many failed orders. Failed count: " + failedOrders);
        }
        
        // 检查是否有可疑的订单模式（简化实现）
        // 实际应该检查订单金额、时间间隔等模式
    }
    
    /**
     * 检查滑点
     */
    public void validateSlippage(BigDecimal expectedRate, BigDecimal actualRate) {
        BigDecimal slippage = actualRate.subtract(expectedRate).abs()
                .divide(expectedRate, 4, java.math.RoundingMode.HALF_UP)
                .multiply(new BigDecimal("100"));
        
        if (slippage.compareTo(maxSlippagePercentage) > 0) {
            throw new RuntimeException("Slippage too high. Expected: " + expectedRate + 
                    ", Actual: " + actualRate + ", Slippage: " + slippage + "%");
        }
    }
    
    /**
     * 转换为USD（简化实现）
     */
    private BigDecimal convertToUSD(BigDecimal amount, String currency) {
        return switch (currency) {
            case "USD" -> amount;
            case "CNY" -> amount.divide(new BigDecimal("7.0"), 2, java.math.RoundingMode.HALF_UP);
            case "EUR" -> amount.multiply(new BigDecimal("1.1"));
            default -> amount;
        };
    }
    
    /**
     * 检查地址风险
     *
     * TODO: 如需接入第三方地址风险识别服务，请在此处实现。
     */
    public void validateWalletAddress(String walletAddress) {
        // 检查是否是黑名单地址
        if (isBlacklistedAddress(walletAddress)) {
            throw new RuntimeException("Wallet address is blacklisted");
        }
        
        // 检查地址格式
        if (!isValidAddressFormat(walletAddress)) {
            throw new RuntimeException("Invalid wallet address format");
        }
    }
    
    /**
     * 检查是否是黑名单地址（简化实现）
     */
    private boolean isBlacklistedAddress(String walletAddress) {
        // 实际应该查询数据库或调用外部服务
        return false;
    }
    
    /**
     * 检查地址格式（简化实现）
     */
    private boolean isValidAddressFormat(String walletAddress) {
        // BTC地址格式检查
        if (walletAddress.matches("^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")) {
            return true;
        }
        
        // ETH地址格式检查
        if (walletAddress.matches("^0x[a-fA-F0-9]{40}$")) {
            return true;
        }
        
        return false;
    }
} 