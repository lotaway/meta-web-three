package com.metawebthree.dataanalysis.application.dto;

import java.util.List;

public class InventoryOverviewDTO {
    private Long totalProducts;
    private Long totalStock;
    private Long lowStockCount;
    private Long overstockCount;
    private Double avgTurnoverRate;
    private List<InventoryAlertDTO> alerts;

    public Long getTotalProducts() {
        return totalProducts;
    }

    public void setTotalProducts(Long totalProducts) {
        this.totalProducts = totalProducts;
    }

    public Long getTotalStock() {
        return totalStock;
    }

    public void setTotalStock(Long totalStock) {
        this.totalStock = totalStock;
    }

    public Long getLowStockCount() {
        return lowStockCount;
    }

    public void setLowStockCount(Long lowStockCount) {
        this.lowStockCount = lowStockCount;
    }

    public Long getOverstockCount() {
        return overstockCount;
    }

    public void setOverstockCount(Long overstockCount) {
        this.overstockCount = overstockCount;
    }

    public Double getAvgTurnoverRate() {
        return avgTurnoverRate;
    }

    public void setAvgTurnoverRate(Double avgTurnoverRate) {
        this.avgTurnoverRate = avgTurnoverRate;
    }

    public List<InventoryAlertDTO> getAlerts() {
        return alerts;
    }

    public void setAlerts(List<InventoryAlertDTO> alerts) {
        this.alerts = alerts;
    }
}