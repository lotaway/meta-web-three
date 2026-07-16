package com.metawebthree.dom.infrastructure.rpc;

public interface InventoryServiceClient {

    Integer checkInventory(String skuCode, Long warehouseId);
}
