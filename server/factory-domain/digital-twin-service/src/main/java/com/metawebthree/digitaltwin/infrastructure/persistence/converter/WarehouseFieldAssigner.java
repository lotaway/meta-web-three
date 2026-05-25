package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WarehouseDO;
import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;

public class WarehouseFieldAssigner {

    public static void assignToEntity(Warehouse warehouse, WarehouseDO warehouseDO) {
        assignBasicFields(warehouse, warehouseDO);
        assignAreaFields(warehouse, warehouseDO);
        assignSpatialFields(warehouse, warehouseDO);
        assignTimestampFields(warehouse, warehouseDO);
    }

    public static void assignToDO(WarehouseDO warehouseDO, Warehouse warehouse) {
        assignIdAndCode(warehouseDO, warehouse);
        assignNameAndDesc(warehouseDO, warehouse);
        assignStatus(warehouseDO, warehouse);
        assignArea(warehouseDO, warehouse);
        assignLocation(warehouseDO, warehouse);
        assignCenter(warehouseDO, warehouse);
        assignDimension(warehouseDO, warehouse);
        assignTimestamp(warehouseDO, warehouse);
    }

    private static void assignIdAndCode(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setId(warehouse.getId());
        warehouseDO.setWarehouseCode(warehouse.getWarehouseCode());
    }

    private static void assignNameAndDesc(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setWarehouseName(warehouse.getWarehouseName());
        warehouseDO.setDescription(warehouse.getDescription());
    }

    private static void assignStatus(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setStatus(warehouse.getStatus() != null ? warehouse.getStatus().name() : null);
    }

    private static void assignArea(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setTotalArea(warehouse.getTotalArea());
        warehouseDO.setUsedArea(warehouse.getUsedArea());
    }

    private static void assignLocation(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setLocation(warehouse.getLocation());
    }

    private static void assignCenter(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setCenterX(warehouse.getCenterX());
        warehouseDO.setCenterY(warehouse.getCenterY());
        warehouseDO.setCenterZ(warehouse.getCenterZ());
    }

    private static void assignDimension(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setWidth(warehouse.getWidth());
        warehouseDO.setLength(warehouse.getLength());
        warehouseDO.setHeight(warehouse.getHeight());
    }

    private static void assignTimestamp(WarehouseDO warehouseDO, Warehouse warehouse) {
        warehouseDO.setCreatedAt(warehouse.getCreatedAt());
        warehouseDO.setUpdatedAt(warehouse.getUpdatedAt());
    }

    private static void assignBasicFields(Warehouse warehouse, WarehouseDO warehouseDO) {
        warehouse.setId(warehouseDO.getId());
        warehouse.setWarehouseCode(warehouseDO.getWarehouseCode());
        warehouse.setWarehouseName(warehouseDO.getWarehouseName());
        warehouse.setDescription(warehouseDO.getDescription());
        warehouse.setStatus(parseStatus(warehouseDO.getStatus()));
    }

    private static void assignAreaFields(Warehouse warehouse, WarehouseDO warehouseDO) {
        warehouse.setTotalArea(warehouseDO.getTotalArea());
        warehouse.setUsedArea(warehouseDO.getUsedArea());
    }

    private static void assignSpatialFields(Warehouse warehouse, WarehouseDO warehouseDO) {
        warehouse.setLocation(warehouseDO.getLocation());
        warehouse.setCenterX(warehouseDO.getCenterX());
        warehouse.setCenterY(warehouseDO.getCenterY());
        warehouse.setCenterZ(warehouseDO.getCenterZ());
        warehouse.setWidth(warehouseDO.getWidth());
        warehouse.setLength(warehouseDO.getLength());
        warehouse.setHeight(warehouseDO.getHeight());
    }

    private static void assignTimestampFields(Warehouse warehouse, WarehouseDO warehouseDO) {
        warehouse.setCreatedAt(warehouseDO.getCreatedAt());
        warehouse.setUpdatedAt(warehouseDO.getUpdatedAt());
    }

    private static WarehouseStatus parseStatus(String status) {
        if (status == null) {
            return null;
        }
        try {
            return WarehouseStatus.valueOf(status);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}