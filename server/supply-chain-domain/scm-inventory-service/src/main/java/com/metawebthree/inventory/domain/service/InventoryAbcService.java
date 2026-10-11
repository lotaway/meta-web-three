package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.AbcClassification;
import java.util.List;

public interface InventoryAbcService {
    List<AbcClassification> classify(Long warehouseId, Integer periodDays);
}