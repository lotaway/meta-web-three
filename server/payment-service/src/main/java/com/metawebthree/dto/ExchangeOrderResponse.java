package com.metawebthree.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import com.metawebthree.common.DO.BaseDO;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
public class ExchangeOrderResponse extends BaseDO {
    
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
    private LocalDateTime expiredAt;
    private String kycLevel;
    private Boolean kycVerified;
    private String remark;
} 