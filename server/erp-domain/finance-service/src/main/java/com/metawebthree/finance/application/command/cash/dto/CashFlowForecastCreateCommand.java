package com.metawebthree.finance.application.command.cash.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

public class CashFlowForecastCreateCommand {
    private String forecastNo;
    private LocalDate forecastDate;
    private LocalDate startDate;
    private LocalDate endDate;
    private String currency;
    private BigDecimal openingBalance;
    private Long createdBy;
    private String creatorName;
    private String remark;
    private List<ForecastItemCreateCommand> items;

    public String getForecastNo() { return forecastNo; }
    public void setForecastNo(String forecastNo) { this.forecastNo = forecastNo; }
    public LocalDate getForecastDate() { return forecastDate; }
    public void setForecastDate(LocalDate forecastDate) { this.forecastDate = forecastDate; }
    public LocalDate getStartDate() { return startDate; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    public LocalDate getEndDate() { return endDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    public BigDecimal getOpeningBalance() { return openingBalance; }
    public void setOpeningBalance(BigDecimal openingBalance) { this.openingBalance = openingBalance; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public List<ForecastItemCreateCommand> getItems() { return items; }
    public void setItems(List<ForecastItemCreateCommand> items) { this.items = items; }

    public static class ForecastItemCreateCommand {
        private String categoryCode;
        private String categoryName;
        private String flowDirection;
        private BigDecimal amount;
        private LocalDate predictedDate;
        private String description;
        private Integer confidenceLevel;
        private String remark;

        public String getCategoryCode() { return categoryCode; }
        public void setCategoryCode(String categoryCode) { this.categoryCode = categoryCode; }
        public String getCategoryName() { return categoryName; }
        public void setCategoryName(String categoryName) { this.categoryName = categoryName; }
        public String getFlowDirection() { return flowDirection; }
        public void setFlowDirection(String flowDirection) { this.flowDirection = flowDirection; }
        public BigDecimal getAmount() { return amount; }
        public void setAmount(BigDecimal amount) { this.amount = amount; }
        public LocalDate getPredictedDate() { return predictedDate; }
        public void setPredictedDate(LocalDate predictedDate) { this.predictedDate = predictedDate; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getConfidenceLevel() { return confidenceLevel; }
        public void setConfidenceLevel(Integer confidenceLevel) { this.confidenceLevel = confidenceLevel; }
        public String getRemark() { return remark; }
        public void setRemark(String remark) { this.remark = remark; }
    }
}