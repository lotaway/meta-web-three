package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.InventoryRecord;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryRecordDO;
import org.springframework.stereotype.Component;

@Component
public class InventoryRecordConverter {

    public InventoryRecord toEntity(InventoryRecordDO doObj) {
        if (doObj == null) {
            return null;
        }
        InventoryRecord entity = new InventoryRecord();
        entity.setId(doObj.getId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setBizType(doObj.getBizType());
        entity.setBizId(doObj.getBizId());
        entity.setQuantity(doObj.getQuantity());
        entity.setBeforeQuantity(doObj.getBeforeQuantity());
        entity.setAfterQuantity(doObj.getAfterQuantity());
        entity.setRemark(doObj.getRemark());
        entity.setOperator(doObj.getOperator());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }

    public InventoryRecordDO toDO(InventoryRecord entity) {
        if (entity == null) {
            return null;
        }
        InventoryRecordDO doObj = new InventoryRecordDO();
        doObj.setId(entity.getId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setBizType(entity.getBizType());
        doObj.setBizId(entity.getBizId());
        doObj.setQuantity(entity.getQuantity());
        doObj.setBeforeQuantity(entity.getBeforeQuantity());
        doObj.setAfterQuantity(entity.getAfterQuantity());
        doObj.setRemark(entity.getRemark());
        doObj.setOperator(entity.getOperator());
        doObj.setCreatedAt(entity.getCreatedAt());
        return doObj;
    }
}