package com.metawebthree.digitaltwin.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("shelves")
public class ShelfDO {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("shelf_code")
    private String shelfCode;

    @TableField("warehouse_code")
    private String warehouseCode;

    @TableField("zone")
    private String zone;

    @TableField("row_number")
    private Integer rowNumber;

    @TableField("column_number")
    private Integer columnNumber;

    @TableField("level_number")
    private Integer levelNumber;

    @TableField("total_levels")
    private Integer totalLevels;

    @TableField("status")
    private String status;

    @TableField("max_weight")
    private BigDecimal maxWeight;

    @TableField("current_weight")
    private BigDecimal currentWeight;

    @TableField("position_x")
    private BigDecimal positionX;

    @TableField("position_y")
    private BigDecimal positionY;

    @TableField("position_z")
    private BigDecimal positionZ;

    @TableField("rotation_y")
    private BigDecimal rotationY;

    @TableField("length")
    private BigDecimal length;

    @TableField("width")
    private BigDecimal width;

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

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
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