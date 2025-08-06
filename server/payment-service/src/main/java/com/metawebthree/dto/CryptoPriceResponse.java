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
public class CryptoPriceResponse {
    
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
    private LocalDateTime updatedAt;
} 