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
    
    @NotNull(message = "订单类型不能为空")
    private String orderType; // BUY_CRYPTO, SELL_CRYPTO
    
    @NotBlank(message = "法币币种不能为空")
    @Pattern(regexp = "^(USD|CNY|EUR)$", message = "不支持的法币币种")
    private String fiatCurrency;
    
    @NotBlank(message = "数字币币种不能为空")
    @Pattern(regexp = "^(BTC|ETH|USDT|USDC)$", message = "不支持的数字币币种")
    private String cryptoCurrency;
    
    @NotNull(message = "金额不能为空")
    @DecimalMin(value = "0.01", message = "金额必须大于0.01")
    @DecimalMax(value = "1000000", message = "金额不能超过1,000,000")
    private BigDecimal amount;
    
    @NotBlank(message = "支付方式不能为空")
    @Pattern(regexp = "^(ALIPAY|WECHAT|BANK_TRANSFER|APPLE_PAY|GOOGLE_PAY)$", message = "不支持的支付方式")
    private String paymentMethod;
    
    @NotBlank(message = "钱包地址不能为空")
    @Pattern(regexp = "^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^0x[a-fA-F0-9]{40}$", message = "无效的钱包地址")
    private String walletAddress;
    
    @Email(message = "邮箱格式不正确")
    private String email;
    
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号格式不正确")
    private String phoneNumber;
    
    private String kycLevel;
    
    private Boolean autoExecute = true; // 是否自动执行
    
    private String remark; // 备注
} 