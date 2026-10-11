package com.metawebthree.inventory.domain.repository.alert;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;

import java.util.List;

public interface InventoryAlertConfigRepository {
    
    List<InventoryAlertConfig> findAll();
    
    List<InventoryAlertConfig> findAllEnabled();
    
    InventoryAlertConfig findById(Long id);
    
    InventoryAlertConfig save(InventoryAlertConfig config);
    
    void deleteById(Long id);
}