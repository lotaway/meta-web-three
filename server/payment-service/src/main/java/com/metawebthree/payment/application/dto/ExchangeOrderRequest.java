package com.metawebthree.payment.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import jakarta.validation.constraints.*;
import java.math.BigDecimal;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Schema(description = "兑换订单请求")
public class ExchangeOrderRequest {
    
    @NotNull(message = "Order type cannot be null")
    @Schema(description = "订单类型: BUY_CRYPTO, SELL_CRYPTO")
    private String orderType;
    
    @NotBlank(message = "Fiat currency cannot be blank")
    @Pattern(regexp = "^(USD|CNY|EUR)$", message = "Unsupported fiat currency")
    @Schema(description = "法币类型")
    private String fiatCurrency;
    
    @NotBlank(message = "Crypto currency cannot be blank")
    @Pattern(regexp = "^(BTC|ETH|USDT|USDC)$", message = "Unsupported crypto currency")
    @Schema(description = "加密货币类型")
    private String cryptoCurrency;
    
    @NotNull(message = "Amount cannot be null")
    @DecimalMin(value = "0.01", message = "Amount must be greater than 0.01")
    @DecimalMax(value = "1000000", message = "Amount cannot exceed 1,000,000")
    @Schema(description = "金额")
    private BigDecimal amount;
    
    @NotBlank(message = "Payment method cannot be blank")
    @Pattern(regexp = "^(ALIPAY|WECHAT|BANK_TRANSFER|APPLE_PAY|GOOGLE_PAY)$", message = "Unsupported payment method")
    @Schema(description = "支付方式")
    private String paymentMethod;
    
    @NotBlank(message = "Wallet address cannot be blank")
    @Pattern(regexp = "^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^0x[a-fA-F0-9]{40}$", message = "Invalid wallet address")
    @Schema(description = "钱包地址")
    private String walletAddress;
    
    @Email(message = "Email format is incorrect")
    @Schema(description = "邮箱")
    private String email;
    
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "Phone number format is incorrect")
    @Schema(description = "电话号码")
    private String phoneNumber;
    
    @Schema(description = "KYC等级")
    private String kycLevel;
    
    @Schema(description = "是否自动执行")
    private Boolean autoExecute = true;
    
    @Schema(description = "备注")
    private String remark;
} 