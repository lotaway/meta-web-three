package com.metawebthree.rma.application.integration;

import java.math.BigDecimal;

/**
 * Integration interface for inventory service.
 * Implementations handle stock-in operations when RMA items are inspected and accepted.
 */
public interface InventoryIntegrationService {

    /**
     * Record accepted returned items back into inventory stock.
     *
     * @param rmaId       RMA order ID
     * @param rmaNo       RMA order number
     * @param skuCode     SKU code of the returned item
     * @param skuName     SKU name of the returned item
     * @param quantity    quantity accepted into stock
     * @param warehouseId warehouse to receive the stock
     */
    void stockIn(Long rmaId, String rmaNo, String skuCode, String skuName, Integer quantity, Long warehouseId);
}
