package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertType;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.repository.InventoryAlertRepository;
import com.metawebthree.digitaltwin.domain.repository.InventoryItemRepository;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.UUID;

@Service
public class InventoryCommandService {

    private static final int ALERT_CODE_LENGTH = 16;
    private final InventoryItemRepository inventoryItemRepository;
    private final InventoryAlertRepository inventoryAlertRepository;
    private final ShelfRepository shelfRepository;
    private final DigitalTwinEventPublisher eventPublisher;

    public InventoryCommandService(InventoryItemRepository inventoryItemRepository,
                                    InventoryAlertRepository inventoryAlertRepository,
                                    ShelfRepository shelfRepository,
                                    DigitalTwinEventPublisher eventPublisher) {
        this.inventoryItemRepository = inventoryItemRepository;
        this.inventoryAlertRepository = inventoryAlertRepository;
        this.shelfRepository = shelfRepository;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public InventoryItem createItem(CreateItemRequest request) {
        validateCreateItemRequest(request);
        InventoryItem item = new InventoryItem(request.itemCode, request.sku, request.itemName);
        assignItemFields(item, request);
        inventoryItemRepository.insert(item);
        return item;
    }

    private void validateCreateItemRequest(CreateItemRequest request) {
        if (inventoryItemRepository.existsByItemCode(request.itemCode)) {
            throw new IllegalArgumentException("Item code already exists: " + request.itemCode);
        }
        if (request.shelfCode != null && !shelfRepository.existsByShelfCode(request.shelfCode)) {
            throw new IllegalArgumentException("Shelf not found: " + request.shelfCode);
        }
    }

    private void assignItemFields(InventoryItem item, CreateItemRequest request) {
        item.setCategory(request.category);
        item.setUnit(request.unit);
        item.setQuantity(request.quantity != null ? request.quantity : BigDecimal.ZERO);
        item.setMinQuantity(request.minQuantity);
        item.setMaxQuantity(request.maxQuantity);
        item.setShelfCode(request.shelfCode);
        item.setBatchNumber(request.batchNumber);
        item.setProductionDate(request.productionDate);
        item.setExpiryDate(request.expiryDate);
        item.setUnitPrice(request.unitPrice);
        item.setLastRestockDate(request.lastRestockDate);
        item.updateStatus();
        item.setCreatedAt(LocalDateTime.now());
        item.setUpdatedAt(LocalDateTime.now());
    }

    @Transactional
    public InventoryItem updateItem(UpdateItemRequest request) {
        InventoryItem item = inventoryItemRepository.findById(request.id)
                .orElseThrow(() -> new IllegalArgumentException("Item not found: " + request.id));
        applyItemUpdates(item, request);
        inventoryItemRepository.update(item);
        return item;
    }

    private void applyItemUpdates(InventoryItem item, UpdateItemRequest request) {
        if (request.itemName != null) {
            item.setItemName(request.itemName);
        }
        if (request.category != null) {
            item.setCategory(request.category);
        }
        if (request.unit != null) {
            item.setUnit(request.unit);
        }
        if (request.minQuantity != null) {
            item.setMinQuantity(request.minQuantity);
        }
        if (request.maxQuantity != null) {
            item.setMaxQuantity(request.maxQuantity);
        }
        if (request.unitPrice != null) {
            item.setUnitPrice(request.unitPrice);
        }
        item.setUpdatedAt(LocalDateTime.now());
        item.updateStatus();
    }

    @Transactional
    public InventoryItem addStock(Long itemId, BigDecimal quantity) {
        InventoryItem item = inventoryItemRepository.findById(itemId)
                .orElseThrow(() -> new IllegalArgumentException("Item not found: " + itemId));
        item.addQuantity(quantity);
        item.setLastRestockDate(LocalDate.now());
        inventoryItemRepository.update(item);
        String warehouseCode = findWarehouseCodeByShelf(item.getShelfCode());
        eventPublisher.publishInventoryLevelChanged(
                warehouseCode,
                item.getSku(),
                item.getQuantity().intValue(),
                item.getStatus().name());
        return item;
    }

    @Transactional
    public InventoryItem removeStock(Long itemId, BigDecimal quantity) {
        InventoryItem item = inventoryItemRepository.findById(itemId)
                .orElseThrow(() -> new IllegalArgumentException("Item not found: " + itemId));
        item.removeQuantity(quantity);
        inventoryItemRepository.update(item);
        String warehouseCode = findWarehouseCodeByShelf(item.getShelfCode());
        eventPublisher.publishInventoryLevelChanged(
                warehouseCode,
                item.getSku(),
                item.getQuantity().intValue(),
                item.getStatus().name());
        checkAndCreateLowStockAlert(item);
        return item;
    }

    @Transactional
    public InventoryAlert acknowledgeAlert(Long alertId, String userId) {
        InventoryAlert alert = inventoryAlertRepository.findById(alertId)
                .orElseThrow(() -> new IllegalArgumentException("Alert not found: " + alertId));
        alert.acknowledge(userId);
        inventoryAlertRepository.update(alert);
        return alert;
    }

    @Transactional
    public InventoryAlert resolveAlert(Long alertId, String userId, String solution) {
        InventoryAlert alert = inventoryAlertRepository.findById(alertId)
                .orElseThrow(() -> new IllegalArgumentException("Alert not found: " + alertId));
        alert.resolve(userId, solution);
        inventoryAlertRepository.update(alert);
        return alert;
    }

    private void checkAndCreateLowStockAlert(InventoryItem item) {
        if (item.needsRestock()) {
            InventoryAlert alert = createLowStockAlert(item);
            inventoryAlertRepository.insert(alert);
            eventPublisher.publishInventoryAlertCreated(
                    alert.getWarehouseCode(),
                    alert.getAlertCode(),
                    alert.getLevel().name(),
                    alert.getDescription());
        }
    }

    private InventoryAlert createLowStockAlert(InventoryItem item) {
        String alertCode = "INV-" + UUID.randomUUID().toString().replace("-", "").substring(0, ALERT_CODE_LENGTH);
        InventoryAlert alert = new InventoryAlert(
                alertCode,
                item.getItemCode(),
                AlertType.LOW_STOCK,
                AlertLevel.WARNING,
                "Low stock alert: " + item.getItemName()
        );
        String warehouseCode = findWarehouseCodeByShelf(item.getShelfCode());
        alert.setWarehouseCode(warehouseCode);
        alert.setShelfCode(item.getShelfCode());
        alert.setCurrentQuantity(item.getQuantity());
        alert.setThresholdValue(item.getMinQuantity());
        alert.setDescription("Current quantity " + item.getQuantity() + " is below minimum " + item.getMinQuantity());
        return alert;
    }

    private String findWarehouseCodeByShelf(String shelfCode) {
        if (shelfCode == null) {
            return null;
        }
        return shelfRepository.findByShelfCode(shelfCode)
                .map(Shelf::getWarehouseCode)
                .orElse(null);
    }

    public static class CreateItemRequest {
        public String itemCode;
        public String sku;
        public String itemName;
        public String category;
        public String unit;
        public BigDecimal quantity;
        public BigDecimal minQuantity;
        public BigDecimal maxQuantity;
        public String shelfCode;
        public String batchNumber;
        public LocalDate productionDate;
        public LocalDate expiryDate;
        public BigDecimal unitPrice;
        public LocalDate lastRestockDate;
    }

    public static class UpdateItemRequest {
        public Long id;
        public String itemName;
        public String category;
        public String unit;
        public BigDecimal minQuantity;
        public BigDecimal maxQuantity;
        public BigDecimal unitPrice;
    }
}