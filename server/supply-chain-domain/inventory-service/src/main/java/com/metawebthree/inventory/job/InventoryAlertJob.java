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
     * 每小时检查一次库存安全预警
     */
    @Scheduled(cron = "0 0 * * * ?")
    public void checkInventorySafetyStock() {
        log.info("开始执行库存安全预警检查任务");
        try {
            List<Inventory> inventories = inventoryRepository.findAll();
            List<InventoryAlertConfig> configs = configRepository.findAllEnabled();
            
            // 批量获取所有 SKU 的最新预警，避免 N+1 查询
            List<String> skuCodes = inventories.stream()
                    .map(Inventory::getSkuCode)
                    .distinct()
                    .collect(Collectors.toList());
            List<InventoryAlert> allLastAlerts = alertRepository.findLastBySkuCodes(skuCodes);
            Map<String, InventoryAlert> lastAlertMap = allLastAlerts.stream()
                    .collect(Collectors.toMap(InventoryAlert::getSkuCode, a -> a, (a, b) -> a));

            for (Inventory inventory : inventories) {
                // 查找对应的预警配置
                InventoryAlertConfig config = findMatchingConfig(inventory, configs);
                
                if (config == null) {
                    // 使用默认配置（如果设置了安全库存）
                    if (inventory.getSafetyStock() != null && inventory.getSafetyStock() > 0) {
                        config = createDefaultConfig(inventory);
                    } else {
                        continue;
                    }
                }

                // 使用预加载的预警数据，避免 N+1 查询
                InventoryAlert lastAlert = lastAlertMap.get(inventory.getSkuCode());
                if (alertDomainService.isInCooldown(lastAlert, config.getCooldownMinutes())) {
                    log.debug("SKU {} 处于预警冷却期内，跳过", inventory.getSkuCode());
                    continue;
                }

                // 检查是否需要生成预警
                List<InventoryAlert> alerts = alertDomainService.checkInventoryAlerts(inventory, config);
                for (InventoryAlert alert : alerts) {
                    alertAppService.createAlert(alert, config);
                }
            }

            log.info("库存安全预警检查任务执行完成");
        } catch (Exception e) {
            log.error("库存安全预警检查任务执行失败: {}", e.getMessage(), e);
        }
    }

    /**
     * 查找匹配的预警配置
     */
    private InventoryAlertConfig findMatchingConfig(Inventory inventory, List<InventoryAlertConfig> configs) {
        for (InventoryAlertConfig config : configs) {
            // 先精确匹配
            if (config.getSkuCode() != null && config.getSkuCode().equals(inventory.getSkuCode())) {
                return config;
            }
        }
        
        // 再按仓库匹配
        for (InventoryAlertConfig config : configs) {
            if (config.getWarehouseCode() != null && 
                config.getWarehouseCode().equals(inventory.getWarehouseId().toString()) &&
                config.getSkuCode() == null) {
                return config;
            }
        }
        
        // 返回全局配置
        for (InventoryAlertConfig config : configs) {
            if (config.getWarehouseCode() == null && config.getSkuCode() == null) {
                return config;
            }
        }
        
        return null;
    }

    /**
     * 创建默认配置
     */
    private InventoryAlertConfig createDefaultConfig(Inventory inventory) {
        InventoryAlertConfig config = new InventoryAlertConfig();
        config.setEnabled(true);
        config.setSafetyStockThreshold(inventory.getSafetyStock());
        config.setLevel(InventoryAlertConfig.AlertLevel.WARNING);
        config.setCooldownMinutes(60); // 默认1小时冷却
        config.setNotificationChannels("EMAIL,IN_APP");
        return config;
    }
}