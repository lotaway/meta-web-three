package com.metawebthree.digitaltwin.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class InventoryItem {
    private static final BigDecimal CRITICAL_THRESHOLD_FACTOR = BigDecimal.valueOf(0.5);

    private Long id;
    private String itemCode;
    private String sku;
    private String itemName;
    private String category;
    private String unit;
    private BigDecimal quantity;
    private BigDecimal minQuantity;
    private BigDecimal maxQuantity;
    private String shelfCode;
    private String batchNumber;
    private LocalDate productionDate;
    private LocalDate expiryDate;
    private BigDecimal unitPrice;
    private ItemStatus status;
    private LocalDate lastRestockDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ItemStatus {
        NORMAL, LOW, CRITICAL, EXPIRED, OUT_OF_STOCK
    }

    public InventoryItem() {
    }

    public InventoryItem(String itemCode, String sku, String itemName) {
        this.itemCode = itemCode;
        this.sku = sku;
        this.itemName = itemName;
        this.quantity = BigDecimal.ZERO;
        this.status = ItemStatus.OUT_OF_STOCK;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addQuantity(BigDecimal amount) {
        if (amount != null && amount.compareTo(BigDecimal.ZERO) > 0) {
            this.quantity = this.quantity.add(amount);
            this.updatedAt = LocalDateTime.now();
            updateStatus();
        }
    }

    public void removeQuantity(BigDecimal amount) {
        if (amount != null && amount.compareTo(BigDecimal.ZERO) > 0) {
            this.quantity = this.quantity.subtract(amount);
            if (this.quantity.compareTo(BigDecimal.ZERO) < 0) {
                this.quantity = BigDecimal.ZERO;
            }
            this.updatedAt = LocalDateTime.now();
            updateStatus();
        }
    }

    public void updateStatus() {
        if (quantity.compareTo(BigDecimal.ZERO) == 0) {
            this.status = ItemStatus.OUT_OF_STOCK;
        } else if (expiryDate != null && expiryDate.isBefore(LocalDate.now())) {
            this.status = ItemStatus.EXPIRED;
        } else if (minQuantity != null && quantity.compareTo(minQuantity) <= 0) {
            if (minQuantity.compareTo(BigDecimal.ZERO) > 0) {
                BigDecimal threshold = minQuantity.multiply(CRITICAL_THRESHOLD_FACTOR);
                this.status = quantity.compareTo(threshold) <= 0 ? ItemStatus.CRITICAL : ItemStatus.LOW;
            } else {
                this.status = ItemStatus.LOW;
            }
        } else {
            this.status = ItemStatus.NORMAL;
        }
    }

    public boolean needsRestock() {
        return minQuantity != null && quantity.compareTo(minQuantity) <= 0;
    }

    public boolean isExpiringSoon(int daysThreshold) {
        if (expiryDate == null) {
            return false;
        }
        return expiryDate.isBefore(LocalDate.now().plusDays(daysThreshold));
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getItemCode() {
        return itemCode;
    }

    public void setItemCode(String itemCode) {
        this.itemCode = itemCode;
    }

    public String getSku() {
        return sku;
    }

    public void setSku(String sku) {
        this.sku = sku;
    }

    public String getItemName() {
        return itemName;
    }

    public void setItemName(String itemName) {
        this.itemName = itemName;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getUnit() {
        return unit;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }

    public BigDecimal getQuantity() {
        return quantity;
    }

    public void setQuantity(BigDecimal quantity) {
        this.quantity = quantity;
    }

    public BigDecimal getMinQuantity() {
        return minQuantity;
    }

    public void setMinQuantity(BigDecimal minQuantity) {
        this.minQuantity = minQuantity;
    }

    public BigDecimal getMaxQuantity() {
        return maxQuantity;
    }

    public void setMaxQuantity(BigDecimal maxQuantity) {
        this.maxQuantity = maxQuantity;
    }

    public String getShelfCode() {
        return shelfCode;
    }

    public void setShelfCode(String shelfCode) {
        this.shelfCode = shelfCode;
    }

    public String getBatchNumber() {
        return batchNumber;
    }

    public void setBatchNumber(String batchNumber) {
        this.batchNumber = batchNumber;
    }

    public LocalDate getProductionDate() {
        return productionDate;
    }

    public void setProductionDate(LocalDate productionDate) {
        this.productionDate = productionDate;
    }

    public LocalDate getExpiryDate() {
        return expiryDate;
    }

    public void setExpiryDate(LocalDate expiryDate) {
        this.expiryDate = expiryDate;
    }

    public BigDecimal getUnitPrice() {
        return unitPrice;
    }

    public void setUnitPrice(BigDecimal unitPrice) {
        this.unitPrice = unitPrice;
    }

    public ItemStatus getStatus() {
        return status;
    }

    public void setStatus(ItemStatus status) {
        this.status = status;
    }

    public LocalDate getLastRestockDate() {
        return lastRestockDate;
    }

    public void setLastRestockDate(LocalDate lastRestockDate) {
        this.lastRestockDate = lastRestockDate;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}