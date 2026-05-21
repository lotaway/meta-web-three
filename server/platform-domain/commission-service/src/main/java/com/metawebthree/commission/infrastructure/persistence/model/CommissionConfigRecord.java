package com.metawebthree.commission.infrastructure.persistence.model;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

@TableName("commission_config")
public class CommissionConfigRecord {
    @TableId(type = IdType.INPUT)
    private Long id;
    private BigDecimal buyRate;
    private String levelRates;
    private Integer maxLevels;
    private Integer returnWindowDays;
    private String confirmMethod;
    private LocalDateTime updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public BigDecimal getBuyRate() { return buyRate; }
    public void setBuyRate(BigDecimal buyRate) { this.buyRate = buyRate; }
    public String getLevelRates() { return levelRates; }
    public void setLevelRates(String levelRates) { this.levelRates = levelRates; }
    public Integer getMaxLevels() { return maxLevels; }
    public void setMaxLevels(Integer maxLevels) { this.maxLevels = maxLevels; }
    public Integer getReturnWindowDays() { return returnWindowDays; }
    public void setReturnWindowDays(Integer returnWindowDays) { this.returnWindowDays = returnWindowDays; }
    public String getConfirmMethod() { return confirmMethod; }
    public void setConfirmMethod(String confirmMethod) { this.confirmMethod = confirmMethod; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
