package com.metawebthree.dto;

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
public class ExchangeOrderRequest {
    
    @NotNull(message = "Order type cannot be null")
    private String orderType; // BUY_CRYPTO, SELL_CRYPTO
    
    @NotBlank(message = "Fiat currency cannot be blank")
    @Pattern(regexp = "^(USD|CNY|EUR)$", message = "Unsupported fiat currency")
    private String fiatCurrency;
    
    @NotBlank(message = "Crypto currency cannot be blank")
    @Pattern(regexp = "^(BTC|ETH|USDT|USDC)$", message = "Unsupported crypto currency")
    private String cryptoCurrency;
    
    @NotNull(message = "Amount cannot be null")
    @DecimalMin(value = "0.01", message = "Amount must be greater than 0.01")
    @DecimalMax(value = "1000000", message = "Amount cannot exceed 1,000,000")
    private BigDecimal amount;
    
    @NotBlank(message = "Payment method cannot be blank")
    @Pattern(regexp = "^(ALIPAY|WECHAT|BANK_TRANSFER|APPLE_PAY|GOOGLE_PAY)$", message = "Unsupported payment method")
    private String paymentMethod;
    
    @NotBlank(message = "Wallet address cannot be blank")
    @Pattern(regexp = "^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^0x[a-fA-F0-9]{40}$", message = "Invalid wallet address")
    private String walletAddress;
    
    @Email(message = "Email format is incorrect")
    private String email;
    
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "Phone number format is incorrect")
    private String phoneNumber;
    
    private String kycLevel;
    
    private Boolean autoExecute = true; // 是否自动执行
    
    private String remark; // 备注
} 