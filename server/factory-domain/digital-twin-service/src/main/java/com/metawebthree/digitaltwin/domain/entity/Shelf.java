package com.metawebthree.digitaltwin.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class Shelf {
    private Long id;
    private String shelfCode;
    private String warehouseCode;
    private String zone;
    private Integer rowNumber;
    private Integer columnNumber;
    private Integer levelNumber;
    private Integer totalLevels;
    private ShelfStatus status;
    private BigDecimal maxWeight;
    private BigDecimal currentWeight;
    private BigDecimal positionX;
    private BigDecimal positionY;
    private BigDecimal positionZ;
    private BigDecimal rotationY;
    private BigDecimal length;
    private BigDecimal width;
    private BigDecimal height;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ShelfStatus {
        EMPTY, OCCUPIED, FULL, MAINTENANCE, OUT_OF_SERVICE
    }

    public Shelf() {
    }

    public Shelf(String shelfCode, String warehouseCode, Integer rowNumber, Integer columnNumber) {
        this.shelfCode = shelfCode;
        this.warehouseCode = warehouseCode;
        this.rowNumber = rowNumber;
        this.columnNumber = columnNumber;
        this.levelNumber = 1;
        this.totalLevels = 3;
        this.status = ShelfStatus.EMPTY;
        this.currentWeight = BigDecimal.ZERO;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void occupy() {
        this.status = ShelfStatus.OCCUPIED;
        this.updatedAt = LocalDateTime.now();
    }

    public void setFull() {
        this.status = ShelfStatus.FULL;
        this.updatedAt = LocalDateTime.now();
    }

    public void clear() {
        this.status = ShelfStatus.EMPTY;
        this.currentWeight = BigDecimal.ZERO;
        this.updatedAt = LocalDateTime.now();
    }

    public void enterMaintenance() {
        this.status = ShelfStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean canAccommodateWeight(BigDecimal weight) {
        if (maxWeight == null || weight == null) {
            return true;
        }
        return currentWeight.add(weight).compareTo(maxWeight) <= 0;
    }

    public void addWeight(BigDecimal weight) {
        if (weight != null) {
            this.currentWeight = this.currentWeight.add(weight);
            this.updatedAt = LocalDateTime.now();
            if (currentWeight.compareTo(maxWeight) >= 0) {
                this.status = ShelfStatus.FULL;
            }
        }
    }

    public void removeWeight(BigDecimal weight) {
        if (weight != null) {
            this.currentWeight = this.currentWeight.subtract(weight);
            if (currentWeight.compareTo(BigDecimal.ZERO) < 0) {
                this.currentWeight = BigDecimal.ZERO;
            }
            this.updatedAt = LocalDateTime.now();
            if (currentWeight.compareTo(BigDecimal.ZERO) == 0) {
                this.status = ShelfStatus.EMPTY;
            } else if (status == ShelfStatus.FULL) {
                this.status = ShelfStatus.OCCUPIED;
            }
        }
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getShelfCode() {
        return shelfCode;
    }

    public void setShelfCode(String shelfCode) {
        this.shelfCode = shelfCode;
    }

    public String getWarehouseCode() {
        return warehouseCode;
    }

    public void setWarehouseCode(String warehouseCode) {
        this.warehouseCode = warehouseCode;
    }

    public String getZone() {
        return zone;
    }

    public void setZone(String zone) {
        this.zone = zone;
    }

    public Integer getRowNumber() {
        return rowNumber;
    }

    public void setRowNumber(Integer rowNumber) {
        this.rowNumber = rowNumber;
    }

    public Integer getColumnNumber() {
        return columnNumber;
    }

    public void setColumnNumber(Integer columnNumber) {
        this.columnNumber = columnNumber;
    }

    public Integer getLevelNumber() {
        return levelNumber;
    }

    public void setLevelNumber(Integer levelNumber) {
        this.levelNumber = levelNumber;
    }

    public Integer getTotalLevels() {
        return totalLevels;
    }

    public void setTotalLevels(Integer totalLevels) {
        this.totalLevels = totalLevels;
    }

    public ShelfStatus getStatus() {
        return status;
    }

    public void setStatus(ShelfStatus status) {
        this.status = status;
    }

    public BigDecimal getMaxWeight() {
        return maxWeight;
    }

    public void setMaxWeight(BigDecimal maxWeight) {
        this.maxWeight = maxWeight;
    }

    public BigDecimal getCurrentWeight() {
        return currentWeight;
    }

    public void setCurrentWeight(BigDecimal currentWeight) {
        this.currentWeight = currentWeight;
    }

    public BigDecimal getPositionX() {
        return positionX;
    }

    public void setPositionX(BigDecimal positionX) {
        this.positionX = positionX;
    }

    public BigDecimal getPositionY() {
        return positionY;
    }

    public void setPositionY(BigDecimal positionY) {
        this.positionY = positionY;
    }

    public BigDecimal getPositionZ() {
        return positionZ;
    }

    public void setPositionZ(BigDecimal positionZ) {
        this.positionZ = positionZ;
    }

    public BigDecimal getRotationY() {
        return rotationY;
    }

    public void setRotationY(BigDecimal rotationY) {
        this.rotationY = rotationY;
    }

    public BigDecimal getLength() {
        return length;
    }

    public void setLength(BigDecimal length) {
        this.length = length;
    }

    public BigDecimal getWidth() {
        return width;
    }

    public void setWidth(BigDecimal width) {
        this.width = width;
    }

    public BigDecimal getHeight() {
        return height;
    }

    public void setHeight(BigDecimal height) {
        this.height = height;
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