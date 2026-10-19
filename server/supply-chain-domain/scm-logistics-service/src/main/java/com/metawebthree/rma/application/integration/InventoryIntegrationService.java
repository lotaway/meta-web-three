package com.metawebthree.rma.application.integration;

/**
 * Records accepted return (RMA) goods back into the inventory service.
 */
public interface InventoryIntegrationService {

    /**
     * Record physically accepted, inspection-passed return goods back into stock.
     *
     * <p>The inventory service is responsible for making this idempotent per RMA
     * line, so a retry of the same line never increases stock twice.</p>
     *
     * @return the integration outcome; never {@code null}
     */
    StockInResult stockIn(Long rmaId, String rmaNo, String skuCode, String skuName,
                          Integer quantity, Long warehouseId);
}
