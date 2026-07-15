package com.metawebthree.dom.infrastructure.persistence.converter;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.infrastructure.persistence.dataobject.DomOrderDO;
import org.springframework.stereotype.Component;

@Component
public class DomOrderConverter {

    public DomOrder toEntity(DomOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        DomOrder entity = new DomOrder();
        entity.setId(doObj.getId());
        entity.setDomOrderNo(doObj.getDomOrderNo());
        entity.setOriginalOrderNo(doObj.getOriginalOrderNo());
        entity.setCustomerId(doObj.getCustomerId());
        entity.setCustomerName(doObj.getCustomerName());
        entity.setStatus(doObj.getStatus());
        entity.setTotalAmount(doObj.getTotalAmount());
        entity.setCurrency(doObj.getCurrency());
        entity.setPriority(doObj.getPriority());
        entity.setSourcingStrategy(doObj.getSourcingStrategy());
        entity.setRegion(doObj.getRegion());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public DomOrderDO toDO(DomOrder entity) {
        if (entity == null) {
            return null;
        }
        DomOrderDO doObj = new DomOrderDO();
        doObj.setId(entity.getId());
        doObj.setDomOrderNo(entity.getDomOrderNo());
        doObj.setOriginalOrderNo(entity.getOriginalOrderNo());
        doObj.setCustomerId(entity.getCustomerId());
        doObj.setCustomerName(entity.getCustomerName());
        doObj.setStatus(entity.getStatus());
        doObj.setTotalAmount(entity.getTotalAmount());
        doObj.setCurrency(entity.getCurrency());
        doObj.setPriority(entity.getPriority());
        doObj.setSourcingStrategy(entity.getSourcingStrategy());
        doObj.setRegion(entity.getRegion());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }
}
