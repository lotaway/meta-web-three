package com.metawebthree.finance.domain.entity.exchange;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRate {
    private Long id;
    private String sourceCurrency;
    private String targetCurrency;
    private BigDecimal rate;
    private LocalDate effectiveDate;
    private ExchangeRateType rateType;
    private Boolean isActive;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ExchangeRateType {
        SPOT, MIDDLE, SELLING, BUYING
    }

    public static ExchangeRate create(String sourceCurrency, String targetCurrency,
            BigDecimal rate, LocalDate effectiveDate, ExchangeRateType rateType, String createdBy) {
        LocalDateTime now = LocalDateTime.now();
        return ExchangeRate.builder()
                .sourceCurrency(sourceCurrency)
                .targetCurrency(targetCurrency)
                .rate(rate)
                .effectiveDate(effectiveDate)
                .rateType(rateType)
                .isActive(true)
                .createdBy(createdBy)
                .createdAt(now)
                .updatedAt(now)
                .build();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateRate(BigDecimal newRate) {
        this.rate = newRate;
        this.updatedAt = LocalDateTime.now();
    }
}