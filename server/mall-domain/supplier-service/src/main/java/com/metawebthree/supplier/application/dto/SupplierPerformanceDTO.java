package com.metawebthree.supplier.application.dto;

public class SupplierPerformanceDTO {

    private Long supplierId;
    private Integer orderCount;
    private Integer onTimeDeliveryRate;
    private Integer qualityScore;
    private Integer responseTime;
    private Integer overallScore;

    public Long getSupplierId() {
        return supplierId;
    }

    public void setSupplierId(Long supplierId) {
        this.supplierId = supplierId;
    }

    public Integer getOrderCount() {
        return orderCount;
    }

    public void setOrderCount(Integer orderCount) {
        this.orderCount = orderCount;
    }

    public Integer getOnTimeDeliveryRate() {
        return onTimeDeliveryRate;
    }

    public void setOnTimeDeliveryRate(Integer onTimeDeliveryRate) {
        this.onTimeDeliveryRate = onTimeDeliveryRate;
    }

    public Integer getQualityScore() {
        return qualityScore;
    }

    public void setQualityScore(Integer qualityScore) {
        this.qualityScore = qualityScore;
    }

    public Integer getResponseTime() {
        return responseTime;
    }

    public void setResponseTime(Integer responseTime) {
        this.responseTime = responseTime;
    }

    public Integer getOverallScore() {
        return overallScore;
    }

    public void setOverallScore(Integer overallScore) {
        this.overallScore = overallScore;
    }
}