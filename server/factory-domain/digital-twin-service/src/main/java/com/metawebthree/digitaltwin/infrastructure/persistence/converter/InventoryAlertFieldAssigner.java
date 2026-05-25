package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryAlertDO;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertStatus;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertType;

public class InventoryAlertFieldAssigner {

    public static void assignToEntity(InventoryAlert alert, InventoryAlertDO alertDO) {
        assignIdAndCode(alert, alertDO);
        assignLocation(alert, alertDO);
        assignAlertInfo(alert, alertDO);
        assignThreshold(alert, alertDO);
        assignStatus(alert, alertDO);
        assignSolution(alert, alertDO);
        assignUserInfo(alert, alertDO);
        assignTimestamp(alert, alertDO);
    }

    public static void assignToDO(InventoryAlertDO alertDO, InventoryAlert alert) {
        assignIdAndCode(alertDO, alert);
        assignLocation(alertDO, alert);
        assignAlertInfo(alertDO, alert);
        assignThreshold(alertDO, alert);
        assignStatus(alertDO, alert);
        assignSolution(alertDO, alert);
        assignUserInfo(alertDO, alert);
        assignTimestamp(alertDO, alert);
    }

    private static void assignIdAndCode(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setId(alertDO.getId());
        alert.setAlertCode(alertDO.getAlertCode());
    }

    private static void assignLocation(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setWarehouseCode(alertDO.getWarehouseCode());
        alert.setShelfCode(alertDO.getShelfCode());
        alert.setItemCode(alertDO.getItemCode());
    }

    private static void assignAlertInfo(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setAlertType(parseAlertType(alertDO.getAlertType()));
        alert.setLevel(parseLevel(alertDO.getLevel()));
        alert.setTitle(alertDO.getTitle());
        alert.setDescription(alertDO.getDescription());
    }

    private static void assignThreshold(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setCurrentQuantity(alertDO.getCurrentQuantity());
        alert.setThresholdValue(alertDO.getThresholdValue());
    }

    private static void assignStatus(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setStatus(parseStatus(alertDO.getStatus()));
    }

    private static void assignSolution(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setSolution(alertDO.getSolution());
    }

    private static void assignUserInfo(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setAcknowledgedBy(alertDO.getAcknowledgedBy());
        alert.setResolvedBy(alertDO.getResolvedBy());
    }

    private static void assignTimestamp(InventoryAlert alert, InventoryAlertDO alertDO) {
        alert.setOccurredAt(alertDO.getOccurredAt());
        alert.setAcknowledgedAt(alertDO.getAcknowledgedAt());
        alert.setResolvedAt(alertDO.getResolvedAt());
        alert.setCreatedAt(alertDO.getCreatedAt());
        alert.setUpdatedAt(alertDO.getUpdatedAt());
    }

    private static void assignIdAndCode(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setId(alert.getId());
        alertDO.setAlertCode(alert.getAlertCode());
    }

    private static void assignLocation(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setWarehouseCode(alert.getWarehouseCode());
        alertDO.setShelfCode(alert.getShelfCode());
        alertDO.setItemCode(alert.getItemCode());
    }

    private static void assignAlertInfo(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setAlertType(alert.getAlertType() != null ? alert.getAlertType().name() : null);
        alertDO.setLevel(alert.getLevel() != null ? alert.getLevel().name() : null);
        alertDO.setTitle(alert.getTitle());
        alertDO.setDescription(alert.getDescription());
    }

    private static void assignThreshold(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setCurrentQuantity(alert.getCurrentQuantity());
        alertDO.setThresholdValue(alert.getThresholdValue());
    }

    private static void assignStatus(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setStatus(alert.getStatus() != null ? alert.getStatus().name() : null);
    }

    private static void assignSolution(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setSolution(alert.getSolution());
    }

    private static void assignUserInfo(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setAcknowledgedBy(alert.getAcknowledgedBy());
        alertDO.setResolvedBy(alert.getResolvedBy());
    }

    private static void assignTimestamp(InventoryAlertDO alertDO, InventoryAlert alert) {
        alertDO.setOccurredAt(alert.getOccurredAt());
        alertDO.setAcknowledgedAt(alert.getAcknowledgedAt());
        alertDO.setResolvedAt(alert.getResolvedAt());
        alertDO.setCreatedAt(alert.getCreatedAt());
        alertDO.setUpdatedAt(alert.getUpdatedAt());
    }

    private static AlertType parseAlertType(String alertType) {
        if (alertType == null) {
            return null;
        }
        try {
            return AlertType.valueOf(alertType);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private static AlertLevel parseLevel(String level) {
        if (level == null) {
            return null;
        }
        try {
            return AlertLevel.valueOf(level);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private static AlertStatus parseStatus(String status) {
        if (status == null) {
            return null;
        }
        try {
            return AlertStatus.valueOf(status);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}