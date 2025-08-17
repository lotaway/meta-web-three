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
@TableName("Crypto_Prices")
public class CryptoPrice {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    @TableField("symbol")
    private String symbol; // BTC-USD, ETH-USD, etc.
    
    @TableField("base_currency")
    private String baseCurrency; // BTC, ETH, USDT
    
    @TableField("quote_currency")
    private String quoteCurrency; // USD, CNY, EUR
    
    @TableField("price")
    private BigDecimal price;
    
    @TableField("bid_price")
    private BigDecimal bidPrice;
    
    @TableField("ask_price")
    private BigDecimal askPrice;
    
    @TableField("volume_24h")
    private BigDecimal volume24h;
    
    @TableField("change_24h")
    private BigDecimal change24h;
    
    @TableField("change_percent_24h")
    private BigDecimal changePercent24h;
    
    @TableField("source")
    private String source; // binance, coinbase, okx
    
    @TableField("timestamp")
    private LocalDateTime timestamp;
    
    @TableField(value = "created_at", fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
} 