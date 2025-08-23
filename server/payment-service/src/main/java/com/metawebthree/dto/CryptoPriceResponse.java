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
public class CryptoPriceResponse extends BaseDO {
    
    private String symbol;
    private String baseCurrency;
    private String quoteCurrency;
    private BigDecimal price;
    private BigDecimal bidPrice;
    private BigDecimal askPrice;
    private BigDecimal volume24h;
    private BigDecimal change24h;
    private BigDecimal changePercent24h;
    private String source;
    private LocalDateTime timestamp;
} 