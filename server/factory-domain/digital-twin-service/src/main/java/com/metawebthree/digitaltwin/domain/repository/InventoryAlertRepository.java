package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertStatus;

import java.util.List;
import java.util.Optional;

public interface InventoryAlertRepository {
    Optional<InventoryAlert> findById(Long id);
    Optional<InventoryAlert> findByAlertCode(String alertCode);
    List<InventoryAlert> findAll();
    List<InventoryAlert> findByWarehouseCode(String warehouseCode);
    List<InventoryAlert> findByItemCode(String itemCode);
    List<InventoryAlert> findByStatus(AlertStatus status);
    List<InventoryAlert> findByLevel(AlertLevel level);
    List<InventoryAlert> findActiveAlerts();
    List<InventoryAlert> findByWarehouseCodeAndStatus(String warehouseCode, AlertStatus status);
    void insert(InventoryAlert inventoryAlert);
    void update(InventoryAlert inventoryAlert);
    void delete(InventoryAlert inventoryAlert);
    boolean existsByAlertCode(String alertCode);
}