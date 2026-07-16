package com.metawebthree.forecasting.domain.entity;

import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
@Table(name = "tb_sales_history")
public class SalesHistory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "sku_code", length = 64)
    private String skuCode;

    @Column(name = "warehouse_id")
    private Long warehouseId;

    @Column(name = "sales_date")
    private LocalDate salesDate;

    @Column(name = "quantity")
    private Integer quantity;

    @Column(name = "sales_channel", length = 32)
    private String salesChannel;

    @Column(name = "created_at")
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

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSkuCode() {
        return skuCode;
    }

    public void setSkuCode(String skuCode) {
        this.skuCode = skuCode;
    }

    public Long getWarehouseId() {
        return warehouseId;
    }

    public void setWarehouseId(Long warehouseId) {
        this.warehouseId = warehouseId;
    }

    public LocalDate getSalesDate() {
        return salesDate;
    }

    public void setSalesDate(LocalDate salesDate) {
        this.salesDate = salesDate;
    }

    public Integer getQuantity() {
        return quantity;
    }

    public void setQuantity(Integer quantity) {
        this.quantity = quantity;
    }

    public String getSalesChannel() {
        return salesChannel;
    }

    public void setSalesChannel(String salesChannel) {
        this.salesChannel = salesChannel;
    }

    public LocalDate getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDate createdAt) {
        this.createdAt = createdAt;
    }
}
