package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WarehouseDO;
import org.springframework.stereotype.Component;

@Component
public class WarehouseConverter {

    public Warehouse toEntity(WarehouseDO warehouseDO) {
        if (warehouseDO == null) {
            return null;
        }
        Warehouse warehouse = new Warehouse();
        WarehouseFieldAssigner.assignToEntity(warehouse, warehouseDO);
        return warehouse;
    }

    public WarehouseDO toDO(Warehouse warehouse) {
        if (warehouse == null) {
            return null;
        }
        WarehouseDO warehouseDO = new WarehouseDO();
        WarehouseFieldAssigner.assignToDO(warehouseDO, warehouse);
        return warehouseDO;
    }
}