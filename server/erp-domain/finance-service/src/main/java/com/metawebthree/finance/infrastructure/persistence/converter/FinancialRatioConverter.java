package com.metawebthree.finance.infrastructure.persistence.converter;

import com.metawebthree.finance.domain.entity.FinancialRatio;
import com.metawebthree.finance.infrastructure.persistence.dataobject.FinancialRatioDO;
import org.springframework.stereotype.Component;

@Component
public class FinancialRatioConverter {

    public FinancialRatio toEntity(FinancialRatioDO doObj) {
        if (doObj == null) {
            return null;
        }
        FinancialRatio entity = new FinancialRatio();
        entity.setId(doObj.getId());
        entity.setRatioType(doObj.getRatioType());
        entity.setValue(doObj.getValue());
        entity.setPeriod(doObj.getPeriod());
        entity.setCalculatedAt(doObj.getCalculatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public FinancialRatioDO toDO(FinancialRatio entity) {
        if (entity == null) {
            return null;
        }
        FinancialRatioDO doObj = new FinancialRatioDO();
        doObj.setId(entity.getId());
        doObj.setRatioType(entity.getRatioType());
        doObj.setValue(entity.getValue());
        doObj.setPeriod(entity.getPeriod());
        doObj.setCalculatedAt(entity.getCalculatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }
}