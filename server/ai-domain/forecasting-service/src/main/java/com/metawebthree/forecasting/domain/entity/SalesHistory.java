package com.metawebthree.forecasting.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import java.time.LocalDate;

@TableName("tb_sales_history")
public class SalesHistory {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("sku_code")
    private String skuCode;

    @TableField("warehouse_id")
    private Long warehouseId;

    @TableField("sales_date")
    private LocalDate salesDate;

    @TableField("quantity")
    private Integer quantity;

    @TableField("sales_channel")
    private String salesChannel;

    @TableField("created_at")
    private LocalDate createdAt;

    public SalesHistory() {
    }

    public SalesHistory(String skuCode, Long warehouseId, LocalDate salesDate, Integer quantity) {
        this.skuCode = skuCode;
        this.warehouseId = warehouseId;
        this.salesDate = salesDate;
        this.quantity = quantity;
        this.createdAt = LocalDate.now();
    }

    public boolean isWithinDays(Integer days) {
        if (days == null || salesDate == null) {
            return false;
        }
        return salesDate.isAfter(LocalDate.now().minusDays(days));
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getSkuCode() { return skuCode; }
    public void setSkuCode(String skuCode) { this.skuCode = skuCode; }
    public Long getWarehouseId() { return warehouseId; }
    public void setWarehouseId(Long warehouseId) { this.warehouseId = warehouseId; }
    public LocalDate getSalesDate() { return salesDate; }
    public void setSalesDate(LocalDate salesDate) { this.salesDate = salesDate; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public String getSalesChannel() { return salesChannel; }
    public void setSalesChannel(String salesChannel) { this.salesChannel = salesChannel; }
    public LocalDate getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDate createdAt) { this.createdAt = createdAt; }
}
