package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.InventoryBatch;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryBatchDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class InventoryBatchConverter {

    public InventoryBatch toEntity(InventoryBatchDO dataObject) {
        if (dataObject == null) {
            return null;
        }
        InventoryBatch entity = new InventoryBatch();
        entity.setId(dataObject.getId());
        entity.setSkuCode(dataObject.getSkuCode());
        entity.setWarehouseId(dataObject.getWarehouseId());
        entity.setBatchNo(dataObject.getBatchNo());
        entity.setQuantity(dataObject.getQuantity());
        entity.setAvailableQuantity(dataObject.getAvailableQuantity());
        entity.setReservedQuantity(dataObject.getReservedQuantity());
        entity.setPickedQuantity(dataObject.getPickedQuantity());
        entity.setInboundDate(dataObject.getInboundDate());
        entity.setProductionDate(dataObject.getProductionDate());
        entity.setExpiryDate(dataObject.getExpiryDate());
        entity.setUnitCost(dataObject.getUnitCost());
        entity.setLocationCode(dataObject.getLocationCode());
        entity.setStatus(dataObject.getStatus());
        entity.setRemark(dataObject.getRemark());
        entity.setCreatedAt(dataObject.getCreatedAt());
        entity.setUpdatedAt(dataObject.getUpdatedAt());
        entity.setVersion(dataObject.getVersion());
        return entity;
    }

    public InventoryBatchDO toDataObject(InventoryBatch entity) {
        if (entity == null) {
            return null;
        }
        InventoryBatchDO dataObject = new InventoryBatchDO();
        dataObject.setId(entity.getId());
        dataObject.setSkuCode(entity.getSkuCode());
        dataObject.setWarehouseId(entity.getWarehouseId());
        dataObject.setBatchNo(entity.getBatchNo());
        dataObject.setQuantity(entity.getQuantity());
        dataObject.setAvailableQuantity(entity.getAvailableQuantity());
        dataObject.setReservedQuantity(entity.getReservedQuantity());
        dataObject.setPickedQuantity(entity.getPickedQuantity());
        dataObject.setInboundDate(entity.getInboundDate());
        dataObject.setProductionDate(entity.getProductionDate());
        dataObject.setExpiryDate(entity.getExpiryDate());
        dataObject.setUnitCost(entity.getUnitCost());
        dataObject.setLocationCode(entity.getLocationCode());
        dataObject.setStatus(entity.getStatus());
        dataObject.setRemark(entity.getRemark());
        dataObject.setCreatedAt(entity.getCreatedAt());
        dataObject.setUpdatedAt(entity.getUpdatedAt());
        dataObject.setVersion(entity.getVersion());
        return dataObject;
    }

    public List<InventoryBatch> toEntityList(List<InventoryBatchDO> dataObjectList) {
        if (dataObjectList == null) {
            return List.of();
        }
        return dataObjectList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}