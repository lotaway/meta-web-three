package com.metawebthree.dom.domain.service;

public interface WarehouseServiceClient {

    WarehouseInfo getWarehouse(Long warehouseId);

    Double getWarehouseDistance(String fromRegion, Long warehouseId);
}
