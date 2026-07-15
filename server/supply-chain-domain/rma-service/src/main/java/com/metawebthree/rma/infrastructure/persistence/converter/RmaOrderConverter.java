package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderDO;
import org.springframework.stereotype.Component;

@Component
public class RmaOrderConverter {

    public RmaOrder toEntity(RmaOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        RmaOrder entity = new RmaOrder();
        entity.setId(doObj.getId());
        entity.setRmaNo(doObj.getRmaNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setReturnType(doObj.getReturnType());
        entity.setStatus(doObj.getStatus());
        entity.setCustomerId(doObj.getCustomerId());
        entity.setCustomerName(doObj.getCustomerName());
        entity.setContactPhone(doObj.getContactPhone());
        entity.setReasonCode(doObj.getReasonCode());
        entity.setReasonDescription(doObj.getReasonDescription());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setTotalQuantity(doObj.getTotalQuantity());
        entity.setTotalAmount(doObj.getTotalAmount());
        entity.setCurrency(doObj.getCurrency());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public RmaOrderDO toDO(RmaOrder entity) {
        if (entity == null) {
            return null;
        }
        RmaOrderDO doObj = new RmaOrderDO();
        doObj.setId(entity.getId());
        doObj.setRmaNo(entity.getRmaNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setReturnType(entity.getReturnType());
        doObj.setStatus(entity.getStatus());
        doObj.setCustomerId(entity.getCustomerId());
        doObj.setCustomerName(entity.getCustomerName());
        doObj.setContactPhone(entity.getContactPhone());
        doObj.setReasonCode(entity.getReasonCode());
        doObj.setReasonDescription(entity.getReasonDescription());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setTotalQuantity(entity.getTotalQuantity());
        doObj.setTotalAmount(entity.getTotalAmount());
        doObj.setCurrency(entity.getCurrency());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }
}
