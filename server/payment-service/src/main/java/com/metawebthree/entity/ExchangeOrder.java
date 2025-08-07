package com.metawebthree.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@TableName("exchange_orders")
public class ExchangeOrder {
    
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
    
    @TableField("exchange_rate")
    private BigDecimal exchangeRate;
    
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
    
    @TableField(value = "created_at", fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
    
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