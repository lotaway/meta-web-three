package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.OutboundStrategy;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.OutboundStrategyDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class OutboundStrategyConverter {

    public OutboundStrategy toEntity(OutboundStrategyDO dataObject) {
        if (dataObject == null) {
            return null;
        }
        OutboundStrategy entity = new OutboundStrategy();
        entity.setId(dataObject.getId());
        entity.setStrategyCode(dataObject.getStrategyCode());
        entity.setStrategyName(dataObject.getStrategyName());
        entity.setStrategyType(dataObject.getStrategyType());
        entity.setWarehouseId(dataObject.getWarehouseId());
        entity.setWarehouseCode(dataObject.getWarehouseCode());
        entity.setSkuCode(dataObject.getSkuCode());
        entity.setSkuCodePattern(dataObject.getSkuCodePattern());
        entity.setPriority(dataObject.getPriority());
        entity.setSpecificBatchNo(dataObject.getSpecificBatchNo());
        entity.setIsActive(dataObject.getIsActive());
        entity.setRemark(dataObject.getRemark());
        entity.setCreator(dataObject.getCreator());
        entity.setCreatedAt(dataObject.getCreatedAt());
        entity.setUpdatedAt(dataObject.getUpdatedAt());
        entity.setVersion(dataObject.getVersion());
        return entity;
    }

    public OutboundStrategyDO toDataObject(OutboundStrategy entity) {
        if (entity == null) {
            return null;
        }
        OutboundStrategyDO dataObject = new OutboundStrategyDO();
        dataObject.setId(entity.getId());
        dataObject.setStrategyCode(entity.getStrategyCode());
        dataObject.setStrategyName(entity.getStrategyName());
        dataObject.setStrategyType(entity.getStrategyType());
        dataObject.setWarehouseId(entity.getWarehouseId());
        dataObject.setWarehouseCode(entity.getWarehouseCode());
        dataObject.setSkuCode(entity.getSkuCode());
        dataObject.setSkuCodePattern(entity.getSkuCodePattern());
        dataObject.setPriority(entity.getPriority());
        dataObject.setSpecificBatchNo(entity.getSpecificBatchNo());
        dataObject.setIsActive(entity.getIsActive());
        dataObject.setRemark(entity.getRemark());
        dataObject.setCreator(entity.getCreator());
        dataObject.setCreatedAt(entity.getCreatedAt());
        dataObject.setUpdatedAt(entity.getUpdatedAt());
        dataObject.setVersion(entity.getVersion());
        return dataObject;
    }

    public List<OutboundStrategy> toEntityList(List<OutboundStrategyDO> dataObjectList) {
        if (dataObjectList == null) {
            return List.of();
        }
        return dataObjectList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}