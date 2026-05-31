package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.ReservationRecord;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReservationRecordDO;
import org.springframework.stereotype.Component;

@Component
public class ReservationRecordConverter {

    public ReservationRecord toEntity(ReservationRecordDO doObj) {
        if (doObj == null) {
            return null;
        }
        ReservationRecord entity = new ReservationRecord();
        entity.setId(doObj.getId());
        entity.setBizId(doObj.getBizId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setQuantity(doObj.getQuantity());
        entity.setStatus(doObj.getStatus());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public ReservationRecordDO toDO(ReservationRecord entity) {
        if (entity == null) {
            return null;
        }
        ReservationRecordDO doObj = new ReservationRecordDO();
        doObj.setId(entity.getId());
        doObj.setBizId(entity.getBizId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setQuantity(entity.getQuantity());
        doObj.setStatus(entity.getStatus());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}