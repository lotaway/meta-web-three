package com.metawebthree.digitaltwin.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class Warehouse {
    private Long id;
    private String warehouseCode;
    private String warehouseName;
    private String description;
    private WarehouseStatus status;
    private BigDecimal totalArea;
    private BigDecimal usedArea;
    private String location;
    private BigDecimal centerX;
    private BigDecimal centerY;
    private BigDecimal centerZ;
    private BigDecimal width;
    private BigDecimal length;
    private BigDecimal height;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum WarehouseStatus {
        PLANNING, CONSTRUCTION, OPERATING, MAINTENANCE, DECOMMISSIONED
    }

    public Warehouse() {
    }

    public Warehouse(String warehouseCode, String warehouseName) {
        this.warehouseCode = warehouseCode;
        this.warehouseName = warehouseName;
        this.status = WarehouseStatus.PLANNING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = WarehouseStatus.OPERATING;
        this.updatedAt = LocalDateTime.now();
    }

    public void enterMaintenance() {
        this.status = WarehouseStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void decommission() {
        this.status = WarehouseStatus.DECOMMISSIONED;
        this.updatedAt = LocalDateTime.now();
    }

    public BigDecimal calculateUtilizationRate() {
        if (totalArea == null || totalArea.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return usedArea.multiply(BigDecimal.valueOf(100))
                .divide(totalArea, 2, BigDecimal.ROUND_HALF_UP);
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getWarehouseCode() {
        return warehouseCode;
    }

    public void setWarehouseCode(String warehouseCode) {
        this.warehouseCode = warehouseCode;
    }

    public String getWarehouseName() {
        return warehouseName;
    }

    public void setWarehouseName(String warehouseName) {
        this.warehouseName = warehouseName;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public WarehouseStatus getStatus() {
        return status;
    }

    public void setStatus(WarehouseStatus status) {
        this.status = status;
    }

    public BigDecimal getTotalArea() {
        return totalArea;
    }

    public void setTotalArea(BigDecimal totalArea) {
        this.totalArea = totalArea;
    }

    public BigDecimal getUsedArea() {
        return usedArea;
    }

    public void setUsedArea(BigDecimal usedArea) {
        this.usedArea = usedArea;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public BigDecimal getCenterX() {
        return centerX;
    }

    public void setCenterX(BigDecimal centerX) {
        this.centerX = centerX;
    }

    public BigDecimal getCenterY() {
        return centerY;
    }

    public void setCenterY(BigDecimal centerY) {
        this.centerY = centerY;
    }

    public BigDecimal getCenterZ() {
        return centerZ;
    }

    public void setCenterZ(BigDecimal centerZ) {
        this.centerZ = centerZ;
    }

    public BigDecimal getWidth() {
        return width;
    }

    public void setWidth(BigDecimal width) {
        this.width = width;
    }

    public BigDecimal getLength() {
        return length;
    }

    public void setLength(BigDecimal length) {
        this.length = length;
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