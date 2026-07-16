package com.metawebthree.dom.infrastructure.persistence.converter;

import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.infrastructure.persistence.dataobject.DomOrderLineDO;
import org.springframework.stereotype.Component;

@Component
public class DomOrderLineConverter {

    public DomOrderLine toEntity(DomOrderLineDO doObj) {
        if (doObj == null) {
            return null;
        }
        DomOrderLine entity = new DomOrderLine();
        entity.setId(doObj.getId());
        entity.setDomOrderId(doObj.getDomOrderId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setSkuName(doObj.getSkuName());
        entity.setQuantity(doObj.getQuantity());
        entity.setFulfilledQuantity(doObj.getFulfilledQuantity());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setWarehouseName(doObj.getWarehouseName());
        entity.setUnitPrice(doObj.getUnitPrice());
        entity.setStatus(doObj.getStatus());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }

    public DomOrderLineDO toDO(DomOrderLine entity) {
        if (entity == null) {
            return null;
        }
        DomOrderLineDO doObj = new DomOrderLineDO();
        doObj.setId(entity.getId());
        doObj.setDomOrderId(entity.getDomOrderId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setSkuName(entity.getSkuName());
        doObj.setQuantity(entity.getQuantity());
        doObj.setFulfilledQuantity(entity.getFulfilledQuantity());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setWarehouseName(entity.getWarehouseName());
        doObj.setUnitPrice(entity.getUnitPrice());
        doObj.setStatus(entity.getStatus());
        doObj.setCreatedAt(entity.getCreatedAt());
        return doObj;
    }
}
