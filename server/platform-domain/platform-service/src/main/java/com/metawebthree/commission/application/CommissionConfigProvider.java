package com.metawebthree.commission.application;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import com.metawebthree.commission.domain.CommissionConfig;
import com.metawebthree.commission.domain.CommissionConfigData;
import com.metawebthree.commission.domain.ports.CommissionConfigStore;
import com.metawebthree.commission.infrastructure.config.CommissionProperties;

public class CommissionConfigProvider {
    private final CommissionConfigStore configStore;
    private final CommissionProperties properties;

    public CommissionConfigProvider(CommissionConfigStore configStore, CommissionProperties properties) {
        this.configStore = configStore;
        this.properties = properties;
    }

    public CommissionConfig getConfig() {
        CommissionConfigData data = configStore.load();
        if (data != null) {
            return validate(toConfig(data));
        }
        return validate(fromProperties());
    }

    private CommissionConfig toConfig(CommissionConfigData data) {
        BigDecimal buyRate = data.getBuyRate();
        List<BigDecimal> levelRates = parseRates(data.getLevelRates());
        int maxLevels = data.getMaxLevels() == null ? levelRates.size() : data.getMaxLevels();
        int returnWindowDays = data.getReturnWindowDays() == null ? 0 : data.getReturnWindowDays();
        return new CommissionConfig(buyRate, levelRates, maxLevels, returnWindowDays);
    }

    private CommissionConfig fromProperties() {
        BigDecimal buyRate = parseRate(properties.getBuyRate());
        List<BigDecimal> levelRates = parseRates(properties.getLevelRates());
        int maxLevels = properties.getMaxLevels() == null ? levelRates.size() : properties.getMaxLevels();
        int returnWindowDays = properties.getReturnWindowDays() == null ? 0 : properties.getReturnWindowDays();
        return new CommissionConfig(buyRate, levelRates, maxLevels, returnWindowDays);
    }

    private BigDecimal parseRate(String value) {
        if (value == null || value.isBlank()) {
            return BigDecimal.ZERO;
        }
        return new BigDecimal(value.trim());
    }

    private List<BigDecimal> parseRates(String value) {
        List<BigDecimal> rates = new ArrayList<>();
        if (value == null || value.isBlank()) {
            return rates;
        }
        String[] parts = value.split(",");
        for (String part : parts) {
            if (!part.isBlank()) {
                rates.add(new BigDecimal(part.trim()));
            }
        }
        return rates;
    }

    private CommissionConfig validate(CommissionConfig config) {
        if (config.getBuyRate() == null || config.getBuyRate().signum() < 0) {
            throw new IllegalStateException("commission buyRate must be non-negative");
        }
        BigDecimal total = BigDecimal.ZERO;
        for (BigDecimal rate : config.getLevelRates()) {
            if (rate.signum() < 0) {
                throw new IllegalStateException("commission levelRate must be non-negative");
            }
            total = total.add(rate);
        }
        if (total.compareTo(BigDecimal.ONE) > 0) {
            throw new IllegalStateException("commission levelRates sum exceeds 1.0");
        }
        return config;
    }
}
