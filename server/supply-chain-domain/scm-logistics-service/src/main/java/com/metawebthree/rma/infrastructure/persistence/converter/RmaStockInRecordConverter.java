package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.RmaStockInRecord;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaStockInRecordDO;
import org.springframework.stereotype.Component;

@Component
public class RmaStockInRecordConverter {

    public RmaStockInRecord toEntity(RmaStockInRecordDO doObj) {
        if (doObj == null) {
            return null;
        }
        RmaStockInRecord entity = new RmaStockInRecord();
        entity.setId(doObj.getId());
        entity.setRmaId(doObj.getRmaId());
        entity.setRmaNo(doObj.getRmaNo());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setSkuName(doObj.getSkuName());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setQuantity(doObj.getQuantity());
        entity.setStatus(doObj.getStatus());
        entity.setRetryCount(doObj.getRetryCount());
        entity.setLastError(doObj.getLastError());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public RmaStockInRecordDO toDO(RmaStockInRecord entity) {
        if (entity == null) {
            return null;
        }
        RmaStockInRecordDO doObj = new RmaStockInRecordDO();
        doObj.setId(entity.getId());
        doObj.setRmaId(entity.getRmaId());
        doObj.setRmaNo(entity.getRmaNo());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setSkuName(entity.getSkuName());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setQuantity(entity.getQuantity());
        doObj.setStatus(entity.getStatus());
        doObj.setRetryCount(entity.getRetryCount());
        doObj.setLastError(entity.getLastError());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
