package com.metawebthree.commission.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "commission")
public class CommissionProperties {
    private String buyRate;
    private String levelRates;
    private Integer maxLevels;
    private Integer returnWindowDays;

    public String getBuyRate() { return buyRate; }
    public void setBuyRate(String buyRate) { this.buyRate = buyRate; }

    public String getLevelRates() { return levelRates; }
    public void setLevelRates(String levelRates) { this.levelRates = levelRates; }

    public Integer getMaxLevels() { return maxLevels; }
    public void setMaxLevels(Integer maxLevels) { this.maxLevels = maxLevels; }

    public Integer getReturnWindowDays() { return returnWindowDays; }
    public void setReturnWindowDays(Integer returnWindowDays) { this.returnWindowDays = returnWindowDays; }
}
