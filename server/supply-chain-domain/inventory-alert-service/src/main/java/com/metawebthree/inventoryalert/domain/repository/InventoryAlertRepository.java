package com.metawebthree.inventoryalert.domain.repository;

import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import java.util.List;

public interface InventoryAlertRepository {
    InventoryAlertDO save(InventoryAlertDO alert);
    InventoryAlertDO findById(Long id);
    List<InventoryAlertDO> findByProductId(Long productId);
    List<InventoryAlertDO> findByWarehouseId(Long warehouseId);
    List<InventoryAlertDO> findByAlertLevel(Integer alertLevel);
    List<InventoryAlertDO> findByStatus(Integer status);
    List<InventoryAlertDO> findAll();
    boolean updateStatus(Long id, Integer status);
    boolean deleteById(Long id);
}