package com.metawebthree.reporting.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("rp_inventory_report")
public class InventoryReportDO {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("report_no")
    private String reportNo;

    @TableField("type")
    private String type;

    @TableField("report_date")
    private LocalDateTime reportDate;

    @TableField("total_inventory_value")
    private BigDecimal totalInventoryValue;

    @TableField("total_sku_count")
    private Integer totalSkuCount;

    @TableField("total_quantity")
    private Integer totalQuantity;

    @TableField("turnover_rate")
    private BigDecimal turnoverRate;

    @TableField("slow_moving_rate")
    private BigDecimal slowMovingRate;

    @TableField("slow_moving_count")
    private Integer slowMovingCount;

    @TableField("warehouse_breakdown")
    private String warehouseBreakdown;

    @TableField("category_breakdown")
    private String categoryBreakdown;

    @TableField("low_stock_items")
    private String lowStockItems;

    @TableField("created_at")
    private LocalDateTime createdAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getReportNo() { return reportNo; }
    public void setReportNo(String reportNo) { this.reportNo = reportNo; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public LocalDateTime getReportDate() { return reportDate; }
    public void setReportDate(LocalDateTime reportDate) { this.reportDate = reportDate; }
    public BigDecimal getTotalInventoryValue() { return totalInventoryValue; }
    public void setTotalInventoryValue(BigDecimal totalInventoryValue) { this.totalInventoryValue = totalInventoryValue; }
    public Integer getTotalSkuCount() { return totalSkuCount; }
    public void setTotalSkuCount(Integer totalSkuCount) { this.totalSkuCount = totalSkuCount; }
    public Integer getTotalQuantity() { return totalQuantity; }
    public void setTotalQuantity(Integer totalQuantity) { this.totalQuantity = totalQuantity; }
    public BigDecimal getTurnoverRate() { return turnoverRate; }
    public void setTurnoverRate(BigDecimal turnoverRate) { this.turnoverRate = turnoverRate; }
    public BigDecimal getSlowMovingRate() { return slowMovingRate; }
    public void setSlowMovingRate(BigDecimal slowMovingRate) { this.slowMovingRate = slowMovingRate; }
    public Integer getSlowMovingCount() { return slowMovingCount; }
    public void setSlowMovingCount(Integer slowMovingCount) { this.slowMovingCount = slowMovingCount; }
    public String getWarehouseBreakdown() { return warehouseBreakdown; }
    public void setWarehouseBreakdown(String warehouseBreakdown) { this.warehouseBreakdown = warehouseBreakdown; }
    public String getCategoryBreakdown() { return categoryBreakdown; }
    public void setCategoryBreakdown(String categoryBreakdown) { this.categoryBreakdown = categoryBreakdown; }
    public String getLowStockItems() { return lowStockItems; }
    public void setLowStockItems(String lowStockItems) { this.lowStockItems = lowStockItems; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}