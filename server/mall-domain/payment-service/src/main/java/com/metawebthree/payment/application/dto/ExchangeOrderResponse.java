package com.metawebthree.payment.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;
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
@Schema(description = "兑换订单响应")
public class ExchangeOrderResponse extends BaseDO {
    
    @Schema(description = "订单号")
    private String orderNo;
    @Schema(description = "状态")
    private String status;
    @Schema(description = "订单类型")
    private String orderType;
    @Schema(description = "法币类型")
    private String fiatCurrency;
    @Schema(description = "加密货币类型")
    private String cryptoCurrency;
    @Schema(description = "法币金额")
    private BigDecimal fiatAmount;
    @Schema(description = "加密货币金额")
    private BigDecimal cryptoAmount;
    @Schema(description = "汇率")
    private BigDecimal exchangeRate;
    @Schema(description = "支付方式")
    private String paymentMethod;
    @Schema(description = "支付链接")
    private String paymentUrl;
    @Schema(description = "二维码数据")
    private String qrCode;
    @Schema(description = "钱包地址")
    private String walletAddress;
    @Schema(description = "过期时间")
    private LocalDateTime expiredAt;
    @Schema(description = "KYC等级")
    private String kycLevel;
    @Schema(description = "KYC是否验证")
    private Boolean kycVerified;
    @Schema(description = "备注")
    private String remark;
} 