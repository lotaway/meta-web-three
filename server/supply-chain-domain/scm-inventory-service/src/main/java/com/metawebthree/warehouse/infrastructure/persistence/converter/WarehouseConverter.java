package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.WarehouseDO;
import org.springframework.stereotype.Component;

@Component
public class WarehouseConverter {

    public Warehouse toEntity(WarehouseDO doObj) {
        if (doObj == null) {
            return null;
        }
        Warehouse entity = new Warehouse();
        entity.setId(doObj.getId());
        entity.setWarehouseCode(doObj.getWarehouseCode());
        entity.setWarehouseName(doObj.getWarehouseName());
        entity.setWarehouseType(doObj.getWarehouseType());
        entity.setProvince(doObj.getProvince());
        entity.setCity(doObj.getCity());
        entity.setDistrict(doObj.getDistrict());
        entity.setAddress(doObj.getAddress());
        entity.setContact(doObj.getContact());
        entity.setPhone(doObj.getPhone());
        entity.setTotalCapacity(doObj.getTotalCapacity());
        entity.setUsedCapacity(doObj.getUsedCapacity());
        entity.setStatus(doObj.getStatus());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setVersion(doObj.getVersion());
        return entity;
    }

    public WarehouseDO toDO(Warehouse entity) {
        if (entity == null) {
            return null;
        }
        WarehouseDO doObj = new WarehouseDO();
        doObj.setId(entity.getId());
        doObj.setWarehouseCode(entity.getWarehouseCode());
        doObj.setWarehouseName(entity.getWarehouseName());
        doObj.setWarehouseType(entity.getWarehouseType());
        doObj.setProvince(entity.getProvince());
        doObj.setCity(entity.getCity());
        doObj.setDistrict(entity.getDistrict());
        doObj.setAddress(entity.getAddress());
        doObj.setContact(entity.getContact());
        doObj.setPhone(entity.getPhone());
        doObj.setTotalCapacity(entity.getTotalCapacity());
        doObj.setUsedCapacity(entity.getUsedCapacity());
        doObj.setAvailableCapacity(entity.getAvailableCapacity());
        doObj.setStatus(entity.getStatus());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setVersion(entity.getVersion());
        return doObj;
    }
}