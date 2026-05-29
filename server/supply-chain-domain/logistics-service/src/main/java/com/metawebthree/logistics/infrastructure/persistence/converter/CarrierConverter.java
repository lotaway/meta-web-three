package com.metawebthree.logistics.infrastructure.persistence.converter;

import com.metawebthree.logistics.domain.entity.Carrier;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.CarrierDO;
import org.springframework.stereotype.Component;

@Component
public class CarrierConverter {

    public Carrier toEntity(CarrierDO doObj) {
        if (doObj == null) {
            return null;
        }
        Carrier entity = new Carrier();
        entity.setId(doObj.getId());
        entity.setCarrierCode(doObj.getCarrierCode());
        entity.setCarrierName(doObj.getCarrierName());
        entity.setCarrierType(doObj.getCarrierType());
        entity.setContact(doObj.getContact());
        entity.setPhone(doObj.getPhone());
        entity.setWebsite(doObj.getWebsite());
        entity.setStatus(doObj.getStatus());
        entity.setBaseFreight(doObj.getBaseFreight());
        entity.setWeightUnitPrice(doObj.getWeightUnitPrice());
        entity.setVolumeUnitPrice(doObj.getVolumeUnitPrice());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public CarrierDO toDO(Carrier entity) {
        if (entity == null) {
            return null;
        }
        CarrierDO doObj = new CarrierDO();
        doObj.setId(entity.getId());
        doObj.setCarrierCode(entity.getCarrierCode());
        doObj.setCarrierName(entity.getCarrierName());
        doObj.setCarrierType(entity.getCarrierType());
        doObj.setContact(entity.getContact());
        doObj.setPhone(entity.getPhone());
        doObj.setWebsite(entity.getWebsite());
        doObj.setStatus(entity.getStatus());
        doObj.setBaseFreight(entity.getBaseFreight());
        doObj.setWeightUnitPrice(entity.getWeightUnitPrice());
        doObj.setVolumeUnitPrice(entity.getVolumeUnitPrice());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}