package com.metawebthree.aiwarehouse.domain.entity;

public enum WarehouseCapability {
    DEMAND_FORECASTING("demand-forecasting", "需求预测", 
        "ai-forecasting", AICapability.FallbackType.ALGORITHM),
    LOCATION_RECOMMENDATION("location-recommendation", "库位推荐",
        "ai-location", AICapability.FallbackType.ALGORITHM),
    RESTOCK_SUGGESTION("restock-suggestion", "补货建议",
        "ai-restock", AICapability.FallbackType.HUMAN),
    ANOMALY_DETECTION("anomaly-detection", "异常检测",
        "ai-anomaly", AICapability.FallbackType.ALGORITHM);

    private final String capabilityId;
    private final String capabilityName;
    private final String serviceName;
    private final AICapability.FallbackType defaultFallbackType;

    WarehouseCapability(String capabilityId, String capabilityName,
            String serviceName, AICapability.FallbackType defaultFallbackType) {
        this.capabilityId = capabilityId;
        this.capabilityName = capabilityName;
        this.serviceName = serviceName;
        this.defaultFallbackType = defaultFallbackType;
    }

    public String getCapabilityId() {
        return capabilityId;
    }

    public String getCapabilityName() {
        return capabilityName;
    }

    public String getServiceName() {
        return serviceName;
    }

    public AICapability.FallbackType getDefaultFallbackType() {
        return defaultFallbackType;
    }
}