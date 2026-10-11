package com.metawebthree.recommendation.domain.aishopping.entity;

import java.time.LocalDateTime;

/**
 * Runtime AI shopping config item (DB override layer, takes precedence over
 * application.yml).
 */
public class AiShoppingConfig {

    private String configKey;
    private String configValue;
    private String description;
    private LocalDateTime updatedAt;

    public AiShoppingConfig() {
    }

    public AiShoppingConfig(String configKey, String configValue, String description) {
        this.configKey = configKey;
        this.configValue = configValue;
        this.description = description;
    }

    public String getConfigKey() {
        return configKey;
    }

    public void setConfigKey(String configKey) {
        this.configKey = configKey;
    }

    public String getConfigValue() {
        return configValue;
    }

    public void setConfigValue(String configValue) {
        this.configValue = configValue;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}
