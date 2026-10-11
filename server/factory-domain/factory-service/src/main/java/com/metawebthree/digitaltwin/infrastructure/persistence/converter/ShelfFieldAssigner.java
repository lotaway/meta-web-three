package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.ShelfDO;
import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ShelfFieldAssigner {
    private static final Logger log = LoggerFactory.getLogger(ShelfFieldAssigner.class);

    public static void assignToEntity(Shelf shelf, ShelfDO shelfDO) {
        assignIdAndCode(shelf, shelfDO);
        assignWarehouseCode(shelf, shelfDO);
        assignPosition(shelf, shelfDO);
        assignStatus(shelf, shelfDO);
        assignWeight(shelf, shelfDO);
        assignDimensions(shelf, shelfDO);
        assignTimestamp(shelf, shelfDO);
    }

    public static void assignToDO(ShelfDO shelfDO, Shelf shelf) {
        assignIdAndCode(shelfDO, shelf);
        assignWarehouseCode(shelfDO, shelf);
        assignPosition(shelfDO, shelf);
        assignStatus(shelfDO, shelf);
        assignWeight(shelfDO, shelf);
        assignDimensions(shelfDO, shelf);
        assignTimestamp(shelfDO, shelf);
    }

    private static void assignIdAndCode(Shelf shelf, ShelfDO shelfDO) {
        shelf.setId(shelfDO.getId());
        shelf.setShelfCode(shelfDO.getShelfCode());
    }

    private static void assignWarehouseCode(Shelf shelf, ShelfDO shelfDO) {
        shelf.setWarehouseCode(shelfDO.getWarehouseCode());
        shelf.setZone(shelfDO.getZone());
        shelf.setRowNumber(shelfDO.getRowNumber());
        shelf.setColumnNumber(shelfDO.getColumnNumber());
        shelf.setLevelNumber(shelfDO.getLevelNumber());
        shelf.setTotalLevels(shelfDO.getTotalLevels());
    }

    private static void assignPosition(Shelf shelf, ShelfDO shelfDO) {
        shelf.setPositionX(shelfDO.getPositionX());
        shelf.setPositionY(shelfDO.getPositionY());
        shelf.setPositionZ(shelfDO.getPositionZ());
        shelf.setRotationY(shelfDO.getRotationY());
    }

    private static void assignStatus(Shelf shelf, ShelfDO shelfDO) {
        shelf.setStatus(parseStatus(shelfDO.getStatus()));
    }

    private static void assignWeight(Shelf shelf, ShelfDO shelfDO) {
        shelf.setMaxWeight(shelfDO.getMaxWeight());
        shelf.setCurrentWeight(shelfDO.getCurrentWeight());
    }

    private static void assignDimensions(Shelf shelf, ShelfDO shelfDO) {
        shelf.setLength(shelfDO.getLength());
        shelf.setWidth(shelfDO.getWidth());
        shelf.setHeight(shelfDO.getHeight());
    }

    private static void assignTimestamp(Shelf shelf, ShelfDO shelfDO) {
        shelf.setCreatedAt(shelfDO.getCreatedAt());
        shelf.setUpdatedAt(shelfDO.getUpdatedAt());
    }

    private static void assignIdAndCode(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setId(shelf.getId());
        shelfDO.setShelfCode(shelf.getShelfCode());
    }

    private static void assignWarehouseCode(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setWarehouseCode(shelf.getWarehouseCode());
        shelfDO.setZone(shelf.getZone());
        shelfDO.setRowNumber(shelf.getRowNumber());
        shelfDO.setColumnNumber(shelf.getColumnNumber());
        shelfDO.setLevelNumber(shelf.getLevelNumber());
        shelfDO.setTotalLevels(shelf.getTotalLevels());
    }

    private static void assignPosition(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setPositionX(shelf.getPositionX());
        shelfDO.setPositionY(shelf.getPositionY());
        shelfDO.setPositionZ(shelf.getPositionZ());
        shelfDO.setRotationY(shelf.getRotationY());
    }

    private static void assignStatus(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setStatus(shelf.getStatus() != null ? shelf.getStatus().name() : null);
    }

    private static void assignWeight(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setMaxWeight(shelf.getMaxWeight());
        shelfDO.setCurrentWeight(shelf.getCurrentWeight());
    }

    private static void assignDimensions(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setLength(shelf.getLength());
        shelfDO.setWidth(shelf.getWidth());
        shelfDO.setHeight(shelf.getHeight());
    }

    private static void assignTimestamp(ShelfDO shelfDO, Shelf shelf) {
        shelfDO.setCreatedAt(shelf.getCreatedAt());
        shelfDO.setUpdatedAt(shelf.getUpdatedAt());
    }

    private static ShelfStatus parseStatus(String status) {
        if (status == null) {
            return null;
        }
        try {
            return ShelfStatus.valueOf(status);
        } catch (IllegalArgumentException e) {
            log.warn("Failed to parse ShelfStatus: invalid value '{}', returning null", status);
            return null;
        }
    }
}