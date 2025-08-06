package com.metawebthree.dto;

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
public class ExchangeOrderResponse {
    
    private String orderNo;
    private String status;
    private String orderType;
    private String fiatCurrency;
    private String cryptoCurrency;
    private BigDecimal fiatAmount;
    private BigDecimal cryptoAmount;
    private BigDecimal exchangeRate;
    private String paymentMethod;
    private String paymentUrl; // 支付链接
    private String qrCode; // 二维码数据
    private String walletAddress;
    private LocalDateTime createdAt;
    private LocalDateTime expiredAt;
    private String kycLevel;
    private Boolean kycVerified;
    private String remark;
} 