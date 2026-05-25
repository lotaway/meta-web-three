package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryItemDO;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem.ItemStatus;

public class InventoryItemFieldAssigner {

    public static void assignToEntity(InventoryItem item, InventoryItemDO itemDO) {
        assignIdAndCode(item, itemDO);
        assignSkuAndName(item, itemDO);
        assignCategoryAndUnit(item, itemDO);
        assignQuantity(item, itemDO);
        assignShelfAndBatch(item, itemDO);
        assignDateInfo(item, itemDO);
        assignPriceAndStatus(item, itemDO);
        assignTimestamp(item, itemDO);
    }

    public static void assignToDO(InventoryItemDO itemDO, InventoryItem item) {
        assignIdAndCode(itemDO, item);
        assignSkuAndName(itemDO, item);
        assignCategoryAndUnit(itemDO, item);
        assignQuantity(itemDO, item);
        assignShelfAndBatch(itemDO, item);
        assignDateInfo(itemDO, item);
        assignPriceAndStatus(itemDO, item);
        assignTimestamp(itemDO, item);
    }

    private static void assignIdAndCode(InventoryItem item, InventoryItemDO itemDO) {
        item.setId(itemDO.getId());
        item.setItemCode(itemDO.getItemCode());
    }

    private static void assignSkuAndName(InventoryItem item, InventoryItemDO itemDO) {
        item.setSku(itemDO.getSku());
        item.setItemName(itemDO.getItemName());
    }

    private static void assignCategoryAndUnit(InventoryItem item, InventoryItemDO itemDO) {
        item.setCategory(itemDO.getCategory());
        item.setUnit(itemDO.getUnit());
    }

    private static void assignQuantity(InventoryItem item, InventoryItemDO itemDO) {
        item.setQuantity(itemDO.getQuantity());
        item.setMinQuantity(itemDO.getMinQuantity());
        item.setMaxQuantity(itemDO.getMaxQuantity());
    }

    private static void assignShelfAndBatch(InventoryItem item, InventoryItemDO itemDO) {
        item.setShelfCode(itemDO.getShelfCode());
        item.setBatchNumber(itemDO.getBatchNumber());
    }

    private static void assignDateInfo(InventoryItem item, InventoryItemDO itemDO) {
        item.setProductionDate(itemDO.getProductionDate());
        item.setExpiryDate(itemDO.getExpiryDate());
    }

    private static void assignPriceAndStatus(InventoryItem item, InventoryItemDO itemDO) {
        item.setUnitPrice(itemDO.getUnitPrice());
        item.setStatus(parseStatus(itemDO.getStatus()));
    }

    private static void assignTimestamp(InventoryItem item, InventoryItemDO itemDO) {
        item.setLastRestockDate(itemDO.getLastRestockDate());
        item.setCreatedAt(itemDO.getCreatedAt());
        item.setUpdatedAt(itemDO.getUpdatedAt());
    }

    private static void assignIdAndCode(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setId(item.getId());
        itemDO.setItemCode(item.getItemCode());
    }

    private static void assignSkuAndName(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setSku(item.getSku());
        itemDO.setItemName(item.getItemName());
    }

    private static void assignCategoryAndUnit(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setCategory(item.getCategory());
        itemDO.setUnit(item.getUnit());
    }

    private static void assignQuantity(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setQuantity(item.getQuantity());
        itemDO.setMinQuantity(item.getMinQuantity());
        itemDO.setMaxQuantity(item.getMaxQuantity());
    }

    private static void assignShelfAndBatch(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setShelfCode(item.getShelfCode());
        itemDO.setBatchNumber(item.getBatchNumber());
    }

    private static void assignDateInfo(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setProductionDate(item.getProductionDate());
        itemDO.setExpiryDate(item.getExpiryDate());
    }

    private static void assignPriceAndStatus(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setUnitPrice(item.getUnitPrice());
        itemDO.setStatus(item.getStatus() != null ? item.getStatus().name() : null);
    }

    private static void assignTimestamp(InventoryItemDO itemDO, InventoryItem item) {
        itemDO.setLastRestockDate(item.getLastRestockDate());
        itemDO.setCreatedAt(item.getCreatedAt());
        itemDO.setUpdatedAt(item.getUpdatedAt());
    }

    private static ItemStatus parseStatus(String status) {
        if (status == null) {
            return null;
        }
        try {
            return ItemStatus.valueOf(status);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}