package com.metawebthree.dom.infrastructure.rpc;

import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;

@Service
public class WarehouseServiceClient {

    private static final Map<Long, WarehouseInfo> WAREHOUSE_MAP = new HashMap<>();

    public WarehouseServiceClient() {
        WAREHOUSE_MAP.put(1L, new WarehouseInfo(1L, "East China Warehouse", "华东", 31.23, 121.47));
        WAREHOUSE_MAP.put(2L, new WarehouseInfo(2L, "North China Warehouse", "华北", 39.90, 116.40));
        WAREHOUSE_MAP.put(3L, new WarehouseInfo(3L, "South China Warehouse", "华南", 23.13, 113.26));
    }

    public WarehouseInfo getWarehouse(Long warehouseId) {
        return WAREHOUSE_MAP.get(warehouseId);
    }

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

    public static class WarehouseInfo {
        private Long id;
        private String name;
        private String region;
        private double latitude;
        private double longitude;

        public WarehouseInfo(Long id, String name, String region, double latitude, double longitude) {
            this.id = id;
            this.name = name;
            this.region = region;
            this.latitude = latitude;
            this.longitude = longitude;
        }

        public Long getId() { return id; }
        public String getName() { return name; }
        public String getRegion() { return region; }
        public double getLatitude() { return latitude; }
        public double getLongitude() { return longitude; }
    }
}
