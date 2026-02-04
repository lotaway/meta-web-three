package com.metawebthree.commission.domain;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

public class CommissionConfig {
    private final BigDecimal buyRate;
    private final List<BigDecimal> levelRates;
    private final int maxLevels;
    private final int returnWindowDays;

    public CommissionConfig(BigDecimal buyRate, List<BigDecimal> levelRates, int maxLevels, int returnWindowDays) {
        this.buyRate = buyRate;
        this.levelRates = levelRates == null ? Collections.emptyList() : List.copyOf(levelRates);
        this.maxLevels = maxLevels;
        this.returnWindowDays = returnWindowDays;
    }

    public BigDecimal getBuyRate() { return buyRate; }
    public List<BigDecimal> getLevelRates() { return levelRates; }
    public int getMaxLevels() { return maxLevels; }
    public int getReturnWindowDays() { return returnWindowDays; }
}
