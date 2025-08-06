package com.metawebthree.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "exchange_orders")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ExchangeOrder {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String orderNo;

    private Long userId;

    private OrderType orderType; // BUY_CRYPTO, SELL_CRYPTO

    private OrderStatus status;

    private String fiatCurrency; // USD, CNY, EUR

    private String cryptoCurrency; // BTC, ETH, USDT, USDC

    private BigDecimal fiatAmount;

    private BigDecimal cryptoAmount;

    private BigDecimal exchangeRate;

    private BigDecimal actualRate;

    private PaymentMethod paymentMethod;

    private String paymentOrderNo;

    private String cryptoTransactionHash;

    private String userWalletAddress;

    private String failureReason;

    private LocalDateTime createdAt;

    private LocalDateTime paidAt;

    private LocalDateTime completedAt;

    private LocalDateTime expiredAt;

    private String kycLevel;

    private Boolean kycVerified;

    public enum OrderType {
        BUY_CRYPTO, // fiat to crypto
        SELL_CRYPTO // crypto to fiat
    }

    public enum OrderStatus {
        PENDING, // waiting for payment
        PAID,
        PROCESSING,
        COMPLETED,
        FAILED,
        EXPIRED,
        CANCELLED,
    }

    public enum PaymentMethod {
        ALIPAY, // Alipay
        WECHAT, // WeChat Pay
        BANK_TRANSFER, // Bank Transfer
        APPLE_PAY, // Apple Pay
        GOOGLE_PAY // Google Pay
    }
}