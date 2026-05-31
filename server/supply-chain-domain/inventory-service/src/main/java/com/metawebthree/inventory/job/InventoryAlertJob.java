package com.metawebthree.inventory.job;

import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import com.metawebthree.inventory.domain.service.InventoryAlertDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertConfigRepository;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertRepository;
import com.metawebthree.inventory.application.InventoryAlertAppService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
@Slf4j
public class InventoryAlertJob {

    private final InventoryRepository inventoryRepository;
    private final InventoryAlertConfigRepository configRepository;
    private final InventoryAlertRepository alertRepository;
    private final InventoryAlertDomainService alertDomainService;
    private final InventoryAlertAppService alertAppService;

    /**
     * Check inventory safety stock alert every hour
     */
    @Scheduled(cron = "0 0 * * * ?")
    public void checkInventorySafetyStock() {
        log.info("Starting inventory safety stock alert check task");
        try {
            List<Inventory> inventories = inventoryRepository.findAll();
            List<InventoryAlertConfig> configs = configRepository.findAllEnabled();
            
            // Batch fetch all SKU's latest alerts to avoid N+1 query
            List<String> skuCodes = inventories.stream()
                    .map(Inventory::getSkuCode)
                    .distinct()
                    .collect(Collectors.toList());
            List<InventoryAlert> allLastAlerts = alertRepository.findLastBySkuCodes(skuCodes);
            Map<String, InventoryAlert> lastAlertMap = allLastAlerts.stream()
                    .collect(Collectors.toMap(InventoryAlert::getSkuCode, a -> a, (a, b) -> a));

            for (Inventory inventory : inventories) {
                // Find matching alert config
                InventoryAlertConfig config = findMatchingConfig(inventory, configs);
                
                if (config == null) {
                    // Use default config (if safety stock is set)
                    if (inventory.getSafetyStock() != null && inventory.getSafetyStock() > 0) {
                        config = createDefaultConfig(inventory);
                    } else {
                        continue;
                    }
                }

                // Use preloaded alert data to avoid N+1 query
                InventoryAlert lastAlert = lastAlertMap.get(inventory.getSkuCode());
                if (alertDomainService.isInCooldown(lastAlert, config.getCooldownMinutes())) {
                    log.debug("SKU {} is in alert cooldown period, skipping", inventory.getSkuCode());
                    continue;
                }

                // Check if alert needs to be generated
                List<InventoryAlert> alerts = alertDomainService.checkInventoryAlerts(inventory, config);
                for (InventoryAlert alert : alerts) {
                    alertAppService.createAlert(alert, config);
                }
            }

            log.info("Inventory safety stock alert check task completed");
        } catch (Exception e) {
            log.error("Inventory safety stock alert check task failed: {}", e.getMessage(), e);
        }
    }

    /**
     * Find matching alert config
     */
    private InventoryAlertConfig findMatchingConfig(Inventory inventory, List<InventoryAlertConfig> configs) {
        for (InventoryAlertConfig config : configs) {
            // First exact match
            if (config.getSkuCode() != null && config.getSkuCode().equals(inventory.getSkuCode())) {
                return config;
            }
        }
        
        // Then match by warehouse
        for (InventoryAlertConfig config : configs) {
            if (config.getWarehouseCode() != null && 
                config.getWarehouseCode().equals(inventory.getWarehouseId().toString()) &&
                config.getSkuCode() == null) {
                return config;
            }
        }
        
        // Return global config
        for (InventoryAlertConfig config : configs) {
            if (config.getWarehouseCode() == null && config.getSkuCode() == null) {
                return config;
            }
        }
        
        return null;
    }

    /**
     * Create default config
     */
    private InventoryAlertConfig createDefaultConfig(Inventory inventory) {
        InventoryAlertConfig config = new InventoryAlertConfig();
        config.setEnabled(true);
        config.setSafetyStockThreshold(inventory.getSafetyStock());
        config.setLevel(InventoryAlertConfig.AlertLevel.WARNING);
        config.setCooldownMinutes(60); // Default 1 hour cooldown
        config.setNotificationChannels("EMAIL,IN_APP");
        return config;
    }
}