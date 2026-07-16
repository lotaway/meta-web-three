package com.metawebthree.dom.infrastructure.rpc;

import com.metawebthree.dom.domain.service.WarehouseInfo;
import com.metawebthree.dom.domain.service.WarehouseServiceClient;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;

@Service
@Primary
public class WarehouseServiceMockClient implements WarehouseServiceClient {

    private static final Map<Long, WarehouseInfo> WAREHOUSE_MAP = new HashMap<>();

    public WarehouseServiceMockClient() {
        WAREHOUSE_MAP.put(1L, new WarehouseInfo(1L, "East China Warehouse", "华东", 31.23, 121.47));
        WAREHOUSE_MAP.put(2L, new WarehouseInfo(2L, "North China Warehouse", "华北", 39.90, 116.40));
        WAREHOUSE_MAP.put(3L, new WarehouseInfo(3L, "South China Warehouse", "华南", 23.13, 113.26));
    }

    @Override
    public WarehouseInfo getWarehouse(Long warehouseId) {
        return WAREHOUSE_MAP.get(warehouseId);
    }

    @Override
    public Double getWarehouseDistance(String fromRegion, Long warehouseId) {
        WarehouseInfo warehouse = WAREHOUSE_MAP.get(warehouseId);
        if (warehouse == null) {
            return 9999.0;
        }
        if (fromRegion == null) {
            return 500.0;
        }
        if (fromRegion.equals(warehouse.getRegion())) {
            return 50.0 + Math.random() * 100;
        }
        return 200.0 + Math.random() * 800;
    }
}
