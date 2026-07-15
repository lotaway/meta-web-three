package com.metawebthree.dom.infrastructure.rpc;

import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;

@Service
public class InventoryServiceClient {

    private final Map<String, Integer> mockInventory = new HashMap<>();

    public InventoryServiceClient() {
        mockInventory.put("SKU001_WH1", 100);
        mockInventory.put("SKU001_WH2", 50);
        mockInventory.put("SKU002_WH1", 200);
        mockInventory.put("SKU002_WH2", 0);
        mockInventory.put("SKU003_WH1", 0);
        mockInventory.put("SKU003_WH2", 75);
    }

    public Integer checkInventory(String skuCode, Long warehouseId) {
        String key = skuCode + "_WH" + warehouseId;
        return mockInventory.getOrDefault(key, 50);
    }
}
