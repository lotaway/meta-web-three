package com.metawebthree.finance.infrastructure.persistence.converter.exchange;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import com.metawebthree.finance.infrastructure.persistence.dataobject.exchange.ExchangeRateDO;
import org.springframework.stereotype.Component;

@Component
public class ExchangeRateConverter {

    public ExchangeRate toEntity(ExchangeRateDO doObj) {
        if (doObj == null) {
            return null;
        }
        return ExchangeRate.builder()
                .id(doObj.getId())
                .sourceCurrency(doObj.getSourceCurrency())
                .targetCurrency(doObj.getTargetCurrency())
                .rate(doObj.getRate())
                .effectiveDate(doObj.getEffectiveDate())
                .rateType(ExchangeRate.ExchangeRateType.valueOf(doObj.getRateType()))
                .isActive(doObj.getIsActive())
                .createdBy(doObj.getCreatedBy())
                .createdAt(doObj.getCreatedAt())
                .updatedAt(doObj.getUpdatedAt())
                .build();
    }

    public ExchangeRateDO toDo(ExchangeRate entity) {
        if (entity == null) {
            return null;
        }
        return ExchangeRateDO.builder()
                .id(entity.getId())
                .sourceCurrency(entity.getSourceCurrency())
                .targetCurrency(entity.getTargetCurrency())
                .rate(entity.getRate())
                .effectiveDate(entity.getEffectiveDate())
                .rateType(entity.getRateType().name())
                .isActive(entity.getIsActive())
                .createdBy(entity.getCreatedBy())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }
}