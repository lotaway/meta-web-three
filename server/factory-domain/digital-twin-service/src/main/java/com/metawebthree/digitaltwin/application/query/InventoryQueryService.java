package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertStatus;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem.ItemStatus;
import com.metawebthree.digitaltwin.domain.repository.InventoryAlertRepository;
import com.metawebthree.digitaltwin.domain.repository.InventoryItemRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class InventoryQueryService {

    private final InventoryItemRepository inventoryItemRepository;
    private final InventoryAlertRepository inventoryAlertRepository;

    public InventoryQueryService(InventoryItemRepository inventoryItemRepository,
                                  InventoryAlertRepository inventoryAlertRepository) {
        this.inventoryItemRepository = inventoryItemRepository;
        this.inventoryAlertRepository = inventoryAlertRepository;
    }

    public Optional<InventoryItem> findItemById(Long id) {
        return inventoryItemRepository.findById(id);
    }

    public Optional<InventoryItem> findItemByItemCode(String itemCode) {
        return inventoryItemRepository.findByItemCode(itemCode);
    }

    public Optional<InventoryItem> findItemBySku(String sku) {
        return inventoryItemRepository.findBySku(sku);
    }

    public List<InventoryItem> findAllItems() {
        return inventoryItemRepository.findAll();
    }

    public List<InventoryItem> findItemsByShelfCode(String shelfCode) {
        return inventoryItemRepository.findByShelfCode(shelfCode);
    }

    public List<InventoryItem> findItemsByStatus(ItemStatus status) {
        return inventoryItemRepository.findByStatus(status);
    }

    public List<InventoryItem> findItemsByCategory(String category) {
        return inventoryItemRepository.findByCategory(category);
    }

    public List<InventoryItem> findLowStockItems() {
        return inventoryItemRepository.findLowStockItems();
    }

    public List<InventoryItem> findExpiringSoonItems(int daysThreshold) {
        return inventoryItemRepository.findExpiringSoonItems(daysThreshold);
    }

    public Optional<InventoryAlert> findAlertById(Long id) {
        return inventoryAlertRepository.findById(id);
    }

    public Optional<InventoryAlert> findAlertByAlertCode(String alertCode) {
        return inventoryAlertRepository.findByAlertCode(alertCode);
    }

    public List<InventoryAlert> findAllAlerts() {
        return inventoryAlertRepository.findAll();
    }

    public List<InventoryAlert> findAlertsByWarehouseCode(String warehouseCode) {
        return inventoryAlertRepository.findByWarehouseCode(warehouseCode);
    }

    public List<InventoryAlert> findAlertsByItemCode(String itemCode) {
        return inventoryAlertRepository.findByItemCode(itemCode);
    }

    public List<InventoryAlert> findAlertsByStatus(AlertStatus status) {
        return inventoryAlertRepository.findByStatus(status);
    }

    public List<InventoryAlert> findActiveAlerts() {
        return inventoryAlertRepository.findActiveAlerts();
    }

    public static class InventoryItemSummary {
        public Long id;
        public String itemCode;
        public String sku;
        public String itemName;
        public String category;
        public String unit;
        public BigDecimal quantity;
        public BigDecimal minQuantity;
        public BigDecimal maxQuantity;
        public String shelfCode;
        public String status;
    }

    public List<InventoryItemSummary> getInventoryItemSummaries() {
        return inventoryItemRepository.findAll().stream()
                .map(item -> {
                    InventoryItemSummary summary = new InventoryItemSummary();
                    summary.id = item.getId();
                    summary.itemCode = item.getItemCode();
                    summary.sku = item.getSku();
                    summary.itemName = item.getItemName();
                    summary.category = item.getCategory();
                    summary.unit = item.getUnit();
                    summary.quantity = item.getQuantity();
                    summary.minQuantity = item.getMinQuantity();
                    summary.maxQuantity = item.getMaxQuantity();
                    summary.shelfCode = item.getShelfCode();
                    summary.status = item.getStatus() != null ? item.getStatus().name() : null;
                    return summary;
                })
                .collect(Collectors.toList());
    }

    public static class InventoryAlertSummary {
        public Long id;
        public String alertCode;
        public String warehouseCode;
        public String shelfCode;
        public String itemCode;
        public String alertType;
        public String level;
        public String title;
        public String description;
        public BigDecimal currentQuantity;
        public BigDecimal thresholdValue;
        public String status;
    }

    public List<InventoryAlertSummary> getInventoryAlertSummaries(String warehouseCode) {
        List<InventoryAlert> alerts = warehouseCode != null
                ? inventoryAlertRepository.findByWarehouseCode(warehouseCode)
                : inventoryAlertRepository.findAll();
        return alerts.stream()
                .map(this::toAlertSummary)
                .collect(Collectors.toList());
    }

    private InventoryAlertSummary toAlertSummary(InventoryAlert alert) {
        InventoryAlertSummary summary = new InventoryAlertSummary();
        summary.id = alert.getId();
        summary.alertCode = alert.getAlertCode();
        summary.warehouseCode = alert.getWarehouseCode();
        summary.shelfCode = alert.getShelfCode();
        summary.itemCode = alert.getItemCode();
        summary.alertType = alert.getAlertType() != null ? alert.getAlertType().name() : null;
        summary.level = alert.getLevel() != null ? alert.getLevel().name() : null;
        summary.title = alert.getTitle();
        summary.description = alert.getDescription();
        summary.currentQuantity = alert.getCurrentQuantity();
        summary.thresholdValue = alert.getThresholdValue();
        summary.status = alert.getStatus() != null ? alert.getStatus().name() : null;
        return summary;
    }
}