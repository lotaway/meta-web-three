package com.metawebthree.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.metawebthree.common.DO.BaseDO;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("Exchange_Orders")
public class ExchangeOrder extends BaseDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("order_no")
    private String orderNo;

    @TableField("user_id")
    private Long userId;

    @TableField("order_type")
    private OrderType orderType; // BUY_CRYPTO, SELL_CRYPTO

    @TableField("status")
    private OrderStatus status;

    @TableField("fiat_currency")
    private String fiatCurrency; // USD, CNY, EUR

    @TableField("crypto_currency")
    private String cryptoCurrency; // BTC, ETH, USDT, USDC

    @TableField("fiat_amount")
    private BigDecimal fiatAmount;

    @TableField("crypto_amount")
    private BigDecimal cryptoAmount;

    private Short cryptoDecimals;

    private BigDecimal fee;

    @TableField("exchange_rate")
    private BigDecimal exchangeRate;

    private BigDecimal settlementAmount;

    @TableField("actual_rate")
    private BigDecimal actualRate;

    @TableField("payment_method")
    private PaymentMethod paymentMethod; // ALIPAY, WECHAT, BANK_TRANSFER

    @TableField("payment_order_no")
    private String paymentOrderNo;

    @TableField("crypto_transaction_hash")
    private String cryptoTransactionHash;

    @TableField("user_wallet_address")
    private String userWalletAddress;

    @TableField("failure_reason")
    private String failureReason;

    @TableField("paid_at")
    private LocalDateTime paidAt;

    @TableField("completed_at")
    private LocalDateTime completedAt;

    @TableField("expired_at")
    private LocalDateTime expiredAt;

    @TableField("kyc_level")
    private String kycLevel;

    @TableField("kyc_verified")
    private Boolean kycVerified;

    @TableField("remark")
    private String remark;

    public enum OrderType {
        BUY_CRYPTO,
        SELL_CRYPTO
    }

    public enum OrderStatus {
        PENDING,
        PAID,
        PROCESSING,
        COMPLETED,
        FAILED,
        EXPIRED,
        CANCELLED
    }

    public enum PaymentMethod {
        ALIPAY,
        WECHAT,
        BANK_TRANSFER,
        APPLE_PAY,
        GOOGLE_PAY
    }
}