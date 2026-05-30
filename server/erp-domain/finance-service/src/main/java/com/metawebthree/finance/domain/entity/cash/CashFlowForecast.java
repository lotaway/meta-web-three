package com.metawebthree.finance.domain.entity.cash;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class CashFlowForecast {
    private Long id;
    private String forecastNo;
    private LocalDate forecastDate;
    private LocalDate startDate;
    private LocalDate endDate;
    private String currency;
    private BigDecimal openingBalance;
    private BigDecimal predictedInflow;
    private BigDecimal predictedOutflow;
    private BigDecimal predictedClosingBalance;
    private String remark;
    private Long createdBy;
    private String creatorName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<ForecastItem> items;

    public void create(String forecastNo, LocalDate forecastDate, LocalDate startDate,
                       LocalDate endDate, String currency, BigDecimal openingBalance,
                       Long createdBy, String creatorName) {
        this.forecastNo = forecastNo;
        this.forecastDate = forecastDate;
        this.startDate = startDate;
        this.endDate = endDate;
        this.currency = currency;
        this.openingBalance = openingBalance;
        this.predictedInflow = BigDecimal.ZERO;
        this.predictedOutflow = BigDecimal.ZERO;
        this.predictedClosingBalance = openingBalance;
        this.createdBy = createdBy;
        this.creatorName = creatorName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.items = new ArrayList<>();
    }

    public void addItem(ForecastItem item) {
        if (items == null) {
            items = new ArrayList<>();
        }
        items.add(item);
        recalculate();
        updatedAt = LocalDateTime.now();
    }

    public void removeItem(Long itemId) {
        if (items == null) {
            return;
        }
        items.removeIf(item -> item.getId().equals(itemId));
        recalculate();
        updatedAt = LocalDateTime.now();
    }

    private void recalculate() {
        if (items == null || items.isEmpty()) {
            predictedInflow = BigDecimal.ZERO;
            predictedOutflow = BigDecimal.ZERO;
            predictedClosingBalance = openingBalance;
            return;
        }
        predictedInflow = items.stream()
                .filter(item -> item.getFlowDirection() == ForecastItem.FlowDirection.INFLOW)
                .map(ForecastItem::getAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        predictedOutflow = items.stream()
                .filter(item -> item.getFlowDirection() == ForecastItem.FlowDirection.OUTFLOW)
                .map(ForecastItem::getAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        predictedClosingBalance = openingBalance.add(predictedInflow).subtract(predictedOutflow);
    }

    public BigDecimal getNetCashFlow() {
        return predictedInflow.subtract(predictedOutflow);
    }

    // Getters and Setters
    public Long getId() { return id; }
    public String getForecastNo() { return forecastNo; }
    public LocalDate getForecastDate() { return forecastDate; }
    public LocalDate getStartDate() { return startDate; }
    public LocalDate getEndDate() { return endDate; }
    public String getCurrency() { return currency; }
    public BigDecimal getOpeningBalance() { return openingBalance; }
    public BigDecimal getPredictedInflow() { return predictedInflow; }
    public BigDecimal getPredictedOutflow() { return predictedOutflow; }
    public BigDecimal getPredictedClosingBalance() { return predictedClosingBalance; }
    public String getRemark() { return remark; }
    public Long getCreatedBy() { return createdBy; }
    public String getCreatorName() { return creatorName; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<ForecastItem> getItems() { return items; }

    public void setId(Long id) { this.id = id; }
    public void setForecastNo(String forecastNo) { this.forecastNo = forecastNo; }
    public void setForecastDate(LocalDate forecastDate) { this.forecastDate = forecastDate; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    public void setCurrency(String currency) { this.currency = currency; }
    public void setOpeningBalance(BigDecimal openingBalance) { this.openingBalance = openingBalance; }
    public void setPredictedInflow(BigDecimal predictedInflow) { this.predictedInflow = predictedInflow; }
    public void setPredictedOutflow(BigDecimal predictedOutflow) { this.predictedOutflow = predictedOutflow; }
    public void setPredictedClosingBalance(BigDecimal predictedClosingBalance) { this.predictedClosingBalance = predictedClosingBalance; }
    public void setRemark(String remark) { this.remark = remark; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public void setItems(List<ForecastItem> items) { this.items = items; }

    public static class ForecastItem {
        private Long id;
        private Long forecastId;
        private String categoryCode;
        private String categoryName;
        private FlowDirection flowDirection;
        private BigDecimal amount;
        private LocalDate predictedDate;
        private String description;
        private Integer confidenceLevel;
        private String remark;

        public enum FlowDirection {
            INFLOW, OUTFLOW
        }

        // Getters and Setters
        public Long getId() { return id; }
        public Long getForecastId() { return forecastId; }
        public String getCategoryCode() { return categoryCode; }
        public String getCategoryName() { return categoryName; }
        public FlowDirection getFlowDirection() { return flowDirection; }
        public BigDecimal getAmount() { return amount; }
        public LocalDate getPredictedDate() { return predictedDate; }
        public String getDescription() { return description; }
        public Integer getConfidenceLevel() { return confidenceLevel; }
        public String getRemark() { return remark; }

        public void setId(Long id) { this.id = id; }
        public void setForecastId(Long forecastId) { this.forecastId = forecastId; }
        public void setCategoryCode(String categoryCode) { this.categoryCode = categoryCode; }
        public void setCategoryName(String categoryName) { this.categoryName = categoryName; }
        public void setFlowDirection(FlowDirection flowDirection) { this.flowDirection = flowDirection; }
        public void setAmount(BigDecimal amount) { this.amount = amount; }
        public void setPredictedDate(LocalDate predictedDate) { this.predictedDate = predictedDate; }
        public void setDescription(String description) { this.description = description; }
        public void setConfidenceLevel(Integer confidenceLevel) { this.confidenceLevel = confidenceLevel; }
        public void setRemark(String remark) { this.remark = remark; }
    }
}