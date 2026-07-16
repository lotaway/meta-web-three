package com.metawebthree.dom.infrastructure.rpc;

public interface WarehouseServiceClient {

    WarehouseInfo getWarehouse(Long warehouseId);

    Double getWarehouseDistance(String fromRegion, Long warehouseId);
}
