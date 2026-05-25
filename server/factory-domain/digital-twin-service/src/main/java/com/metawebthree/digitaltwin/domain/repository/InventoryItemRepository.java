package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem.ItemStatus;

import java.util.List;
import java.util.Optional;

public interface InventoryItemRepository {
    Optional<InventoryItem> findById(Long id);
    Optional<InventoryItem> findByItemCode(String itemCode);
    Optional<InventoryItem> findBySku(String sku);
    List<InventoryItem> findAll();
    List<InventoryItem> findByShelfCode(String shelfCode);
    List<InventoryItem> findByStatus(ItemStatus status);
    List<InventoryItem> findByCategory(String category);
    List<InventoryItem> findLowStockItems();
    List<InventoryItem> findExpiringSoonItems(int daysThreshold);
    InventoryItem save(InventoryItem inventoryItem);
    void delete(InventoryItem inventoryItem);
    boolean existsByItemCode(String itemCode);
    boolean existsBySku(String sku);
}