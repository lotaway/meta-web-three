package com.metawebthree.rma.infrastructure.integration;

import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.stereotype.Component;

/**
 * Stub implementation of InventoryIntegrationService.
 * Logs integration points. Replace with real RPC/HTTP client when inventory service is available.
 */
@Slf4j
@Component
@ConditionalOnMissingBean(InventoryIntegrationService.class)
public class StubInventoryIntegrationService implements InventoryIntegrationService {

    @Override
    public void stockIn(Long rmaId, String rmaNo, String skuCode, String skuName, Integer quantity, Long warehouseId) {
        log.info("[INTEGRATION-STUB] Inventory stockIn called — would record accepted return items back into stock");
        log.info("  rmaId={}, rmaNo={}, skuCode={}, skuName={}, quantity={}, warehouseId={}",
                rmaId, rmaNo, skuCode, skuName, quantity, warehouseId);
    }
}
