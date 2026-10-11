package com.metawebthree.warehouse.domain.service;

import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.domain.entity.Location;
import java.util.Optional;

public interface WarehouseDomainService {

    Optional<Warehouse> findById(Long id);

    Warehouse create(String warehouseCode, String warehouseName, String warehouseType);

    void updateCapacity(Warehouse warehouse, Integer quantity, boolean increase);

    Optional<Location> findLocationById(Long locationId);

    Location createLocation(Long warehouseId, String zoneCode, String shelfCode,
            Integer row, Integer column, Integer layer);

    void occupyLocation(Location location);

    void releaseLocation(Location location);
}