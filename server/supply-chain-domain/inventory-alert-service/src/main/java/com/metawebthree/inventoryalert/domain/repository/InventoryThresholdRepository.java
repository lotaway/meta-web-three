package com.metawebthree.inventoryalert.domain.repository;

import com.metawebthree.inventoryalert.domain.model.InventoryThresholdDO;
import java.util.List;

public interface InventoryThresholdRepository {
    InventoryThresholdDO save(InventoryThresholdDO threshold);
    InventoryThresholdDO findById(Long id);
    InventoryThresholdDO findBySkuId(Long skuId);
    List<InventoryThresholdDO> findAll();
    boolean deleteById(Long id);
}