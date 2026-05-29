package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryDO;
import org.springframework.stereotype.Component;

@Component
public class InventoryConverter {

    public Inventory toEntity(InventoryDO doObj) {
        if (doObj == null) {
            return null;
        }
        Inventory entity = new Inventory();
        entity.setId(doObj.getId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setTotalQuantity(doObj.getTotalQuantity());
        entity.setAvailableQuantity(doObj.getAvailableQuantity());
        entity.setReservedQuantity(doObj.getReservedQuantity());
        entity.setDefectiveQuantity(doObj.getDefectiveQuantity());
        entity.setUnitCost(doObj.getUnitCost());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public InventoryDO toDO(Inventory entity) {
        if (entity == null) {
            return null;
        }
        InventoryDO doObj = new InventoryDO();
        doObj.setId(entity.getId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setTotalQuantity(entity.getTotalQuantity());
        doObj.setAvailableQuantity(entity.getAvailableQuantity());
        doObj.setReservedQuantity(entity.getReservedQuantity());
        doObj.setDefectiveQuantity(entity.getDefectiveQuantity());
        doObj.setUnitCost(entity.getUnitCost());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }
}