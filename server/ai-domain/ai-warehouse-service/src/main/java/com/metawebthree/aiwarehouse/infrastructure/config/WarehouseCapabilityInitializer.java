package com.metawebthree.aiwarehouse.infrastructure.config;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.domain.repository.AICapabilityRepository;
import com.metawebthree.aiwarehouse.domain.service.AIWarehouseDomainService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.Map;

@Component
public class WarehouseCapabilityInitializer {

    private static final Logger log = LoggerFactory.getLogger(
        WarehouseCapabilityInitializer.class);

    private final AIWarehouseDomainService domainService;
    private final AICapabilityRepository capabilityRepository;
    private final Map<WarehouseCapability, CapabilityConfig> configs;

    public WarehouseCapabilityInitializer(
            AIWarehouseDomainService domainService,
            AICapabilityRepository capabilityRepository) {
        this.domainService = domainService;
        this.capabilityRepository = capabilityRepository;
        this.configs = initConfigs();
    }

    private Map<WarehouseCapability, CapabilityConfig> initConfigs() {
        Map<WarehouseCapability, CapabilityConfig> map = new EnumMap<>(
            WarehouseCapability.class);
        
        map.put(WarehouseCapability.DEMAND_FORECASTING,
            new CapabilityConfig("forecasting", "DEMAND_FORECASTING", "ALGORITHM"));
        map.put(WarehouseCapability.LOCATION_RECOMMENDATION,
            new CapabilityConfig("location", "LOCATION_RECOMMENDATION", "ALGORITHM"));
        map.put(WarehouseCapability.RESTOCK_SUGGESTION,
            new CapabilityConfig("restock", "RESTOCK_SUGGESTION", "HUMAN"));
        map.put(WarehouseCapability.ANOMALY_DETECTION,
            new CapabilityConfig("anomaly", "ANOMALY_DETECTION", "ALGORITHM"));
        
        return map;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void registerWarehouseCapabilities() {
        log.info("Starting warehouse capability registration...");
        
        for (WarehouseCapability capability : WarehouseCapability.values()) {
            registerCapability(capability);
        }
        
        log.info("Warehouse capability registration completed");
    }

    private void registerCapability(WarehouseCapability capability) {
        try {
            if (capabilityRepository.existsByCapabilityId(capability.getCapabilityId())) {
                log.debug("Capability already exists, skipping: {}",
                    capability.getCapabilityId());
                return;
            }
            
            CapabilityConfig config = configs.get(capability);
            
            domainService.registerCapability(
                capability.getCapabilityId(),
                capability.getCapabilityName(),
                AICapability.AICapabilityType.valueOf(config.type),
                "http://" + config.serviceName + "/api/v1/predict",
                AICapability.FallbackType.valueOf(config.fallbackType),
                buildFallbackConfig(capability)
            );
            
            log.info("Registered warehouse capability: {} -> {}",
                capability.getCapabilityId(), config.serviceName);
        } catch (Exception e) {
            log.error("Failed to register capability {}: {}",
                capability.getCapabilityId(), e.getMessage());
        }
    }

    private String buildFallbackConfig(WarehouseCapability capability) {
        return String.format(
            "{\"capability\":\"%s\",\"algorithmTimeout\":5000,"
            + "\"humanTicketPriority\":\"NORMAL\",\"maxRetries\":3}",
            capability.name()
        );
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