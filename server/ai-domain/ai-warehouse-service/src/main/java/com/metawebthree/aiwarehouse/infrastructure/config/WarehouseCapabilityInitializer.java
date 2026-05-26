package com.metawebthree.aiwarehouse.infrastructure.config;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.domain.repository.AICapabilityRepository;
import com.metawebthree.aiwarehouse.domain.service.IAIWarehouseDomainService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.Map;

@Component
public class WarehouseCapabilityInitializer {

    private static final Logger log = LoggerFactory.getLogger(
        WarehouseCapabilityInitializer.class);
    private static final int DEFAULT_TIMEOUT_MS = 5000;
    private static final int DEFAULT_MAX_RETRIES = 3;
    private static final String DEFAULT_PROTOCOL = "https://";
    private static final String DEFAULT_API_PATH = "/api/v1/predict";

    private final IAIWarehouseDomainService domainService;
    private final AICapabilityRepository capabilityRepository;
    private final Map<WarehouseCapability, CapabilityConfig> configs;

    @Value("${warehouse.ai.base-url:localhost:8080}")
    private String aiServiceBaseUrl;

    public WarehouseCapabilityInitializer(
            IAIWarehouseDomainService domainService,
            AICapabilityRepository capabilityRepository) {
        this.domainService = domainService;
        this.capabilityRepository = capabilityRepository;
        this.configs = initConfigs();
    }

    private Map<WarehouseCapability, CapabilityConfig> initConfigs() {
        Map<WarehouseCapability, CapabilityConfig> map = new EnumMap<>(
            WarehouseCapability.class);
        
        map.put(WarehouseCapability.DEMAND_FORECASTING,
            new CapabilityConfig("forecasting", "FORECASTING", "ALGORITHM"));
        map.put(WarehouseCapability.LOCATION_RECOMMENDATION,
            new CapabilityConfig("location", "RECOMMENDATION", "ALGORITHM"));
        map.put(WarehouseCapability.RESTOCK_SUGGESTION,
            new CapabilityConfig("restock", "RECOMMENDATION", "HUMAN"));
        map.put(WarehouseCapability.ANOMALY_DETECTION,
            new CapabilityConfig("anomaly", "RISK_SCORING", "ALGORITHM"));
        
        return map;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void registerWarehouseCapabilities() {
        log.info("Starting warehouse capability registration...");
        
        for (WarehouseCapability capability : WarehouseCapability.values()) {
            boolean success = registerCapability(capability);
            if (!success) {
                log.warn("Capability registration failed, will retry on next startup: {}",
                    capability.getCapabilityId());
            }
        }
        
        log.info("Warehouse capability registration completed");
    }

    private boolean registerCapability(WarehouseCapability capability) {
        try {
            if (checkCapabilityExists(capability)) {
                return true;
            }
            doRegisterCapability(capability);
            return true;
        } catch (Exception e) {
            log.error("Failed to register capability {}: {}",
                capability.getCapabilityId(), e.getMessage(), e);
            return false;
        }
    }

    private boolean checkCapabilityExists(WarehouseCapability capability) {
        if (capabilityRepository.existsByCapabilityId(capability.getCapabilityId())) {
            log.info("Capability already exists, skipping: {}",
                capability.getCapabilityId());
            return true;
        }
        return false;
    }

    private void doRegisterCapability(WarehouseCapability capability) {
        CapabilityConfig config = configs.get(capability);
        String endpointUrl = buildEndpointUrl(config.serviceName);
        
        domainService.registerCapability(
            capability.getCapabilityId(),
            capability.getCapabilityName(),
            AICapability.AICapabilityType.valueOf(config.type),
            endpointUrl,
            AICapability.FallbackType.valueOf(config.fallbackType),
            buildFallbackConfig(capability)
        );
        
        log.info("Registered warehouse capability: {} -> {}",
            capability.getCapabilityId(), config.serviceName);
    }

    private String buildEndpointUrl(String serviceName) {
        return DEFAULT_PROTOCOL + aiServiceBaseUrl + DEFAULT_API_PATH;
    }

    private String buildFallbackConfig(WarehouseCapability capability) {
        return String.format(
            "{\"capability\":\"%s\",\"algorithmTimeout\":%d,"
            + "\"humanTicketPriority\":\"NORMAL\",\"maxRetries\":%d}",
            capability.name(), DEFAULT_TIMEOUT_MS, DEFAULT_MAX_RETRIES);
    }

    private static class CapabilityConfig {
        final String serviceName;
        final String type;
        final String fallbackType;

        CapabilityConfig(String serviceName, String type, String fallbackType) {
            this.serviceName = serviceName;
            this.type = type;
            this.fallbackType = fallbackType;
        }
    }
}