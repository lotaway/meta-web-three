package com.metawebthree.finance.infrastructure.persistence.dataobject.cash;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.math.BigDecimal;
import java.time.LocalDate;

@TableName("cash_flow_forecast_item")
public class CashFlowForecastItemDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long forecastId;
    private String categoryCode;
    private String categoryName;
    private String flowDirection;
    private BigDecimal amount;
    private LocalDate predictedDate;
    private String description;
    private String confidenceLevel;
    private String remark;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getForecastId() { return forecastId; }
    public void setForecastId(Long forecastId) { this.forecastId = forecastId; }
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
    public String getConfidenceLevel() { return confidenceLevel; }
    public void setConfidenceLevel(String confidenceLevel) { this.confidenceLevel = confidenceLevel; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
}