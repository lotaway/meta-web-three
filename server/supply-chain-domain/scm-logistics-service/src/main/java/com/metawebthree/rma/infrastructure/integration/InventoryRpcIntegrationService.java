package com.metawebthree.rma.infrastructure.integration;

import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import com.metawebthree.rma.application.integration.StockInResult;
import com.metawebthree.supplychain.generated.rpc.InventoryService;
import com.metawebthree.supplychain.generated.rpc.StockInRequest;
import com.metawebthree.supplychain.generated.rpc.StockInResponse;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

/**
 * Real {@link InventoryIntegrationService} that records accepted return goods
 * through the inventory service {@code stockIn} RPC.
 *
 * <p>Failures are returned as a {@link StockInResult} (not swallowed) so the
 * caller can persist the failed attempt and retry it later.</p>
 */
@Slf4j
@Component
public class InventoryRpcIntegrationService implements InventoryIntegrationService {

    @DubboReference(check = false, lazy = true)
    private InventoryService inventoryService;

    @Override
    public StockInResult stockIn(Long rmaId, String rmaNo, String skuCode, String skuName,
                                 Integer quantity, Long warehouseId) {
        try {
            StockInResponse response = inventoryService.stockIn(StockInRequest.newBuilder()
                    .setRmaId(rmaId != null ? rmaId : 0L)
                    .setRmaNo(rmaNo != null ? rmaNo : "")
                    .setSkuCode(skuCode != null ? skuCode : "")
                    .setQuantity(quantity != null ? quantity : 0)
                    .setWarehouseId(warehouseId != null ? warehouseId : 0L)
                    .build());
            if (response.getSuccess()) {
                log.info("RMA stock-in succeeded - rmaNo={}, skuCode={}, quantity={}, duplicated={}",
                        rmaNo, skuCode, quantity, response.getDuplicated());
                return StockInResult.success(response.getDuplicated());
            }
            log.warn("RMA stock-in rejected - rmaNo={}, skuCode={}, message={}",
                    rmaNo, skuCode, response.getMessage());
            return StockInResult.failure(response.getMessage());
        } catch (Exception e) {
            log.error("RMA stock-in RPC failed - rmaNo={}, skuCode={}, error={}",
                    rmaNo, skuCode, e.getMessage(), e);
            return StockInResult.failure("Inventory service unavailable: " + e.getMessage());
        }
    }
}
