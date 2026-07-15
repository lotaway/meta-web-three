package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderItemDO;
import org.springframework.stereotype.Component;

@Component
public class RmaOrderItemConverter {

    public RmaOrderItem toEntity(RmaOrderItemDO doObj) {
        if (doObj == null) {
            return null;
        }
        RmaOrderItem entity = new RmaOrderItem();
        entity.setId(doObj.getId());
        entity.setRmaId(doObj.getRmaId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setSkuName(doObj.getSkuName());
        entity.setExpectedQuantity(doObj.getExpectedQuantity());
        entity.setInspectedQuantity(doObj.getInspectedQuantity());
        entity.setAcceptedQuantity(doObj.getAcceptedQuantity());
        entity.setUnitPrice(doObj.getUnitPrice());
        entity.setReasonCode(doObj.getReasonCode());
        entity.setReasonDescription(doObj.getReasonDescription());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }

    public RmaOrderItemDO toDO(RmaOrderItem entity) {
        if (entity == null) {
            return null;
        }
        RmaOrderItemDO doObj = new RmaOrderItemDO();
        doObj.setId(entity.getId());
        doObj.setRmaId(entity.getRmaId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setSkuName(entity.getSkuName());
        doObj.setExpectedQuantity(entity.getExpectedQuantity());
        doObj.setInspectedQuantity(entity.getInspectedQuantity());
        doObj.setAcceptedQuantity(entity.getAcceptedQuantity());
        doObj.setUnitPrice(entity.getUnitPrice());
        doObj.setReasonCode(entity.getReasonCode());
        doObj.setReasonDescription(entity.getReasonDescription());
        doObj.setCreatedAt(entity.getCreatedAt());
        return doObj;
    }
}
