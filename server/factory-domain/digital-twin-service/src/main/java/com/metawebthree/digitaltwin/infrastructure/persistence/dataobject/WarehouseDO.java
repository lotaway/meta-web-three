package com.metawebthree.digitaltwin.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("warehouses")
public class WarehouseDO {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("warehouse_code")
    private String warehouseCode;

    @TableField("warehouse_name")
    private String warehouseName;

    @TableField("description")
    private String description;

    @TableField("status")
    private String status;

    @TableField("total_area")
    private BigDecimal totalArea;

    @TableField("used_area")
    private BigDecimal usedArea;

    @TableField("location")
    private String location;

    @TableField("center_x")
    private BigDecimal centerX;

    @TableField("center_y")
    private BigDecimal centerY;

    @TableField("center_z")
    private BigDecimal centerZ;

    @TableField("width")
    private BigDecimal width;

    @TableField("length")
    private BigDecimal length;

    @TableField("height")
    private BigDecimal height;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;

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

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
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