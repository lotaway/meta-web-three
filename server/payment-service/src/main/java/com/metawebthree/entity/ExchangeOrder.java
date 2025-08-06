package com.metawebthree.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "exchange_orders")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ExchangeOrder {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(unique = true, nullable = false)
    private String orderNo;
    
    @Column(nullable = false)
    private Long userId;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private OrderType orderType; // BUY_CRYPTO, SELL_CRYPTO
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private OrderStatus status;
    
    @Column(nullable = false)
    private String fiatCurrency; // USD, CNY, EUR
    
    @Column(nullable = false)
    private String cryptoCurrency; // BTC, ETH, USDT, USDC
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal fiatAmount;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal cryptoAmount;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal exchangeRate;
    
    @Column(precision = 20, scale = 8)
    private BigDecimal actualRate;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private PaymentMethod paymentMethod; // ALIPAY, WECHAT, BANK_TRANSFER
    
    @Column
    private String paymentOrderNo;
    
    @Column
    private String cryptoTransactionHash;
    
    @Column
    private String userWalletAddress;
    
    @Column
    private String failureReason;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @Column
    private LocalDateTime paidAt;
    
    @Column
    private LocalDateTime completedAt;
    
    @Column
    private LocalDateTime expiredAt;
    
    @Column
    private String kycLevel;
    
    @Column
    private Boolean kycVerified;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (status == null) {
            status = OrderStatus.PENDING;
        }
    }
    
    public enum OrderType {
        BUY_CRYPTO,    // 法币购买数字币
        SELL_CRYPTO    // 数字币兑换法币
    }
    
    public enum OrderStatus {
        PENDING,        // 待支付
        PAID,          // 已支付
        PROCESSING,    // 处理中
        COMPLETED,     // 已完成
        FAILED,        // 失败
        EXPIRED,       // 已过期
        CANCELLED      // 已取消
    }
    
    public enum PaymentMethod {
        ALIPAY,        // 支付宝
        WECHAT,        // 微信支付
        BANK_TRANSFER, // 银行转账
        APPLE_PAY,     // Apple Pay
        GOOGLE_PAY     // Google Pay
    }
} 