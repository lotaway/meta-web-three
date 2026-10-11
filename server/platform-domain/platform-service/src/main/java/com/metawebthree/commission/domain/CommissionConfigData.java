package com.metawebthree.commission.domain;

import java.math.BigDecimal;

public class CommissionConfigData {
    private BigDecimal buyRate;
    private String levelRates;
    private Integer maxLevels;
    private Integer returnWindowDays;

    public BigDecimal getBuyRate() { return buyRate; }
    public void setBuyRate(BigDecimal buyRate) { this.buyRate = buyRate; }
    public String getLevelRates() { return levelRates; }
    public void setLevelRates(String levelRates) { this.levelRates = levelRates; }
    public Integer getMaxLevels() { return maxLevels; }
    public void setMaxLevels(Integer maxLevels) { this.maxLevels = maxLevels; }
    public Integer getReturnWindowDays() { return returnWindowDays; }
    public void setReturnWindowDays(Integer returnWindowDays) { this.returnWindowDays = returnWindowDays; }
}
