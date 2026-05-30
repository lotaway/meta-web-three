package com.metawebthree.finance.infrastructure.persistence.dataobject.exchange;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRateDO {
    private Long id;
    private String sourceCurrency;
    private String targetCurrency;
    private BigDecimal rate;
    private LocalDate effectiveDate;
    private String rateType;
    private Boolean isActive;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}