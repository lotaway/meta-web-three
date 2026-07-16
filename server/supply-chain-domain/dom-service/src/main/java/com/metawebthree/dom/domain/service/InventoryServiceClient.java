package com.metawebthree.dom.domain.service;

public interface InventoryServiceClient {

    Integer checkInventory(String skuCode, Long warehouseId);
}
