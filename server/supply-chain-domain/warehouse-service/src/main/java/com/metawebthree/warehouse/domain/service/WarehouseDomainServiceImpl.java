package com.metawebthree.warehouse.domain.service;

import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.domain.entity.Location;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class WarehouseDomainServiceImpl implements WarehouseDomainService {

    @Override
    public Optional<Warehouse> findById(Long id) {
        return Optional.empty();
    }

    @Override
    public Warehouse create(String warehouseCode, String warehouseName, String warehouseType) {
        Warehouse warehouse = new Warehouse();
        warehouse.setWarehouseCode(warehouseCode);
        warehouse.setWarehouseName(warehouseName);
        warehouse.setWarehouseType(warehouseType);
        warehouse.setStatus("ACTIVE");
        warehouse.setTotalCapacity(0);
        warehouse.setUsedCapacity(0);
        return warehouse;
    }

    @Override
    public void updateCapacity(Warehouse warehouse, Integer quantity, boolean increase) {
        if (increase) {
            warehouse.increaseCapacity(quantity);
        } else {
            warehouse.decreaseCapacity(quantity);
        }
    }

    @Override
    public Optional<Location> findLocationById(Long locationId) {
        return Optional.empty();
    }

    @Override
    public Location createLocation(Long warehouseId, String zoneCode, String shelfCode,
            Integer row, Integer column, Integer layer) {
        Location location = new Location();
        location.setWarehouseId(warehouseId);
        location.setZoneCode(zoneCode);
        location.setShelfCode(shelfCode);
        location.setRow(row);
        location.setColumn(column);
        location.setLayer(layer);
        location.setStatus("IDLE");
        location.setLocationCode(String.format("%s-%s-%d-%d-%d",
            zoneCode, shelfCode, row, column, layer));
        return location;
    }

    @Override
    public void occupyLocation(Location location) {
        location.occupy();
    }

    @Override
    public void releaseLocation(Location location) {
        location.release();
    }
}