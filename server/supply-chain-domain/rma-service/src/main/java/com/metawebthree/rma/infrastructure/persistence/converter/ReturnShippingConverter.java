package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.ReturnShipping;
import com.metawebthree.rma.infrastructure.persistence.dataobject.ReturnShippingDO;
import org.springframework.stereotype.Component;

@Component
public class ReturnShippingConverter {

    public ReturnShipping toEntity(ReturnShippingDO doObj) {
        if (doObj == null) {
            return null;
        }
        ReturnShipping entity = new ReturnShipping();
        entity.setId(doObj.getId());
        entity.setRmaId(doObj.getRmaId());
        entity.setRmaNo(doObj.getRmaNo());
        entity.setCarrier(doObj.getCarrier());
        entity.setTrackingNo(doObj.getTrackingNo());
        entity.setShippingMethod(doObj.getShippingMethod());
        entity.setOriginAddress(doObj.getOriginAddress());
        entity.setDestinationAddress(doObj.getDestinationAddress());
        entity.setShippingDate(doObj.getShippingDate());
        entity.setEstimatedArrivalDate(doObj.getEstimatedArrivalDate());
        entity.setStatus(doObj.getStatus());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public ReturnShippingDO toDO(ReturnShipping entity) {
        if (entity == null) {
            return null;
        }
        ReturnShippingDO doObj = new ReturnShippingDO();
        doObj.setId(entity.getId());
        doObj.setRmaId(entity.getRmaId());
        doObj.setRmaNo(entity.getRmaNo());
        doObj.setCarrier(entity.getCarrier());
        doObj.setTrackingNo(entity.getTrackingNo());
        doObj.setShippingMethod(entity.getShippingMethod());
        doObj.setOriginAddress(entity.getOriginAddress());
        doObj.setDestinationAddress(entity.getDestinationAddress());
        doObj.setShippingDate(entity.getShippingDate());
        doObj.setEstimatedArrivalDate(entity.getEstimatedArrivalDate());
        doObj.setStatus(entity.getStatus());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
