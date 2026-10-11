package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryAlertDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class InventoryAlertConverter {

    public InventoryAlert toEntity(InventoryAlertDO doObj) {
        if (doObj == null) {
            return null;
        }
        InventoryAlert entity = new InventoryAlert();
        entity.setId(doObj.getId());
        entity.setAlertCode(doObj.getAlertCode());
        entity.setWarehouseCode(doObj.getWarehouseCode());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setAlertType(InventoryAlert.AlertType.valueOf(doObj.getAlertType()));
        entity.setLevel(InventoryAlert.AlertLevel.valueOf(doObj.getLevel()));
        entity.setTitle(doObj.getTitle());
        entity.setDescription(doObj.getDescription());
        entity.setCurrentQuantity(doObj.getCurrentQuantity());
        entity.setThresholdValue(doObj.getThresholdValue());
        entity.setStatus(InventoryAlert.AlertStatus.valueOf(doObj.getStatus()));
        entity.setSolution(doObj.getSolution());
        entity.setAcknowledgedBy(doObj.getAcknowledgedBy());
        entity.setAcknowledgedAt(doObj.getAcknowledgedAt());
        entity.setResolvedBy(doObj.getResolvedBy());
        entity.setResolvedAt(doObj.getResolvedAt());
        entity.setOccurredAt(doObj.getOccurredAt());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public InventoryAlertDO toDO(InventoryAlert entity) {
        if (entity == null) {
            return null;
        }
        InventoryAlertDO doObj = new InventoryAlertDO();
        doObj.setId(entity.getId());
        doObj.setAlertCode(entity.getAlertCode());
        doObj.setWarehouseCode(entity.getWarehouseCode());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setAlertType(entity.getAlertType() != null ? entity.getAlertType().name() : null);
        doObj.setLevel(entity.getLevel() != null ? entity.getLevel().name() : null);
        doObj.setTitle(entity.getTitle());
        doObj.setDescription(entity.getDescription());
        doObj.setCurrentQuantity(entity.getCurrentQuantity());
        doObj.setThresholdValue(entity.getThresholdValue());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setSolution(entity.getSolution());
        doObj.setAcknowledgedBy(entity.getAcknowledgedBy());
        doObj.setAcknowledgedAt(entity.getAcknowledgedAt());
        doObj.setResolvedBy(entity.getResolvedBy());
        doObj.setResolvedAt(entity.getResolvedAt());
        doObj.setOccurredAt(entity.getOccurredAt());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }

    public List<InventoryAlert> toEntityList(List<InventoryAlertDO> doList) {
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
}