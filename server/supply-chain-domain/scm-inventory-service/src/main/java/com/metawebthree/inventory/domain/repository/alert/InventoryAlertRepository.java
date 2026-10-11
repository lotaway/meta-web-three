package com.metawebthree.inventory.domain.repository.alert;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;

import java.util.List;

public interface InventoryAlertRepository {
    
    List<InventoryAlert> findAll();
    
    List<InventoryAlert> findActiveAlerts();
    
    InventoryAlert findById(Long id);
    
    InventoryAlert findByAlertCode(String alertCode);
    
    InventoryAlert findLastBySkuCode(String skuCode);
    
    List<InventoryAlert> findBySkuCodeAndStatus(String skuCode, InventoryAlert.AlertStatus status);
    
    InventoryAlert save(InventoryAlert alert);
    
    void deleteById(Long id);
    
    long countBySkuCodeAndStatus(String skuCode, InventoryAlert.AlertStatus status);
    
    List<InventoryAlert> findLastBySkuCodes(List<String> skuCodes);
}