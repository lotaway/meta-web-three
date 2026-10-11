package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaDispositionDO;
import org.springframework.stereotype.Component;

@Component
public class RmaDispositionConverter {

    public RmaDisposition toEntity(RmaDispositionDO doObj) {
        if (doObj == null) {
            return null;
        }
        RmaDisposition entity = new RmaDisposition();
        entity.setId(doObj.getId());
        entity.setRmaId(doObj.getRmaId());
        entity.setRmaNo(doObj.getRmaNo());
        entity.setDispositionType(doObj.getDispositionType());
        entity.setRefundAmount(doObj.getRefundAmount());
        entity.setReplacementSkuCode(doObj.getReplacementSkuCode());
        entity.setReplacementQuantity(doObj.getReplacementQuantity());
        entity.setScrapQuantity(doObj.getScrapQuantity());
        entity.setScrapReason(doObj.getScrapReason());
        entity.setDispositionBy(doObj.getDispositionBy());
        entity.setDispositionDate(doObj.getDispositionDate());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public RmaDispositionDO toDO(RmaDisposition entity) {
        if (entity == null) {
            return null;
        }
        RmaDispositionDO doObj = new RmaDispositionDO();
        doObj.setId(entity.getId());
        doObj.setRmaId(entity.getRmaId());
        doObj.setRmaNo(entity.getRmaNo());
        doObj.setDispositionType(entity.getDispositionType());
        doObj.setRefundAmount(entity.getRefundAmount());
        doObj.setReplacementSkuCode(entity.getReplacementSkuCode());
        doObj.setReplacementQuantity(entity.getReplacementQuantity());
        doObj.setScrapQuantity(entity.getScrapQuantity());
        doObj.setScrapReason(entity.getScrapReason());
        doObj.setDispositionBy(entity.getDispositionBy());
        doObj.setDispositionDate(entity.getDispositionDate());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
