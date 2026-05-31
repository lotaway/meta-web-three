package com.metawebthree.dataanalysis.application.dto;

import java.util.List;

public class SalesTrendDTO {
    private List<SalesStatisticsDTO> trendList;
    private Long totalOrders;
    private Long totalAmount;
    private Long totalNewUsers;
    private Double growthRate;

    public List<SalesStatisticsDTO> getTrendList() {
        return trendList;
    }

    public void setTrendList(List<SalesStatisticsDTO> trendList) {
        this.trendList = trendList;
    }

    public Long getTotalOrders() {
        return totalOrders;
    }

    public void setTotalOrders(Long totalOrders) {
        this.totalOrders = totalOrders;
    }

    public Long getTotalAmount() {
        return totalAmount;
    }

    public void setTotalAmount(Long totalAmount) {
        this.totalAmount = totalAmount;
    }

    public Long getTotalNewUsers() {
        return totalNewUsers;
    }

    public void setTotalNewUsers(Long totalNewUsers) {
        this.totalNewUsers = totalNewUsers;
    }

    public Double getGrowthRate() {
        return growthRate;
    }

    public void setGrowthRate(Double growthRate) {
        this.growthRate = growthRate;
    }
}