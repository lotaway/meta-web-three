package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryAlertConfigDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class InventoryAlertConfigConverter {

    public InventoryAlertConfig toEntity(InventoryAlertConfigDO doObj) {
        if (doObj == null) {
            return null;
        }
        InventoryAlertConfig entity = new InventoryAlertConfig();
        entity.setId(doObj.getId());
        entity.setConfigCode(doObj.getConfigCode());
        entity.setWarehouseCode(doObj.getWarehouseCode());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setSafetyStockThreshold(doObj.getSafetyStockThreshold());
        entity.setLevel(InventoryAlertConfig.AlertLevel.valueOf(doObj.getLevel()));
        entity.setEnabled(doObj.getEnabled());
        entity.setCooldownMinutes(doObj.getCooldownMinutes());
        entity.setNotificationChannels(doObj.getNotificationChannels());
        entity.setNotifyUsers(doObj.getNotifyUsers());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedBy(doObj.getUpdatedBy());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public InventoryAlertConfigDO toDO(InventoryAlertConfig entity) {
        if (entity == null) {
            return null;
        }
        InventoryAlertConfigDO doObj = new InventoryAlertConfigDO();
        doObj.setId(entity.getId());
        doObj.setConfigCode(entity.getConfigCode());
        doObj.setWarehouseCode(entity.getWarehouseCode());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setSafetyStockThreshold(entity.getSafetyStockThreshold());
        doObj.setLevel(entity.getLevel() != null ? entity.getLevel().name() : null);
        doObj.setEnabled(entity.getEnabled());
        doObj.setCooldownMinutes(entity.getCooldownMinutes());
        doObj.setNotificationChannels(entity.getNotificationChannels());
        doObj.setNotifyUsers(entity.getNotifyUsers());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedBy(entity.getUpdatedBy());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }

    public List<InventoryAlertConfig> toEntityList(List<InventoryAlertConfigDO> doList) {
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
}