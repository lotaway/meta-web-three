package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.InboundOrder;
import com.metawebthree.warehouse.domain.entity.InboundOrderItem;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.InboundOrderDO;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.InboundOrderItemDO;
import org.springframework.stereotype.Component;


@Component
public class InboundOrderConverter {

    public InboundOrder toEntity(InboundOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        InboundOrder entity = new InboundOrder();
        entity.setId(doObj.getId());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setInboundType(doObj.getInboundType());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setSupplierCode(doObj.getSupplierCode());
        entity.setStatus(doObj.getStatus());
        entity.setRemark(doObj.getRemark());
        entity.setOperator(doObj.getOperator());
        entity.setPlanArrivalTime(doObj.getPlanArrivalTime());
        entity.setActualArrivalTime(doObj.getActualArrivalTime());
        entity.setCompletedAt(doObj.getCompletedAt());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public InboundOrderDO toDO(InboundOrder entity) {
        if (entity == null) {
            return null;
        }
        InboundOrderDO doObj = new InboundOrderDO();
        doObj.setId(entity.getId());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setInboundType(entity.getInboundType());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setSupplierCode(entity.getSupplierCode());
        doObj.setStatus(entity.getStatus());
        doObj.setRemark(entity.getRemark());
        doObj.setOperator(entity.getOperator());
        doObj.setPlanArrivalTime(entity.getPlanArrivalTime());
        doObj.setActualArrivalTime(entity.getActualArrivalTime());
        doObj.setCompletedAt(entity.getCompletedAt());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    public InboundOrderItem toEntityItem(InboundOrderItemDO doObj) {
        if (doObj == null) {
            return null;
        }
        InboundOrderItem entity = new InboundOrderItem();
        entity.setId(doObj.getId());
        entity.setOrderId(doObj.getOrderId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setProductName(doObj.getProductName());
        entity.setPlanQuantity(doObj.getPlanQuantity());
        entity.setActualQuantity(doObj.getActualQuantity());
        entity.setLocationId(doObj.getLocationId());
        entity.setStatus(doObj.getStatus());
        entity.setUnitCost(doObj.getUnitCost());
        entity.setBatchNo(doObj.getBatchNo());
        entity.setProductionDate(doObj.getProductionDate());
        entity.setExpiryDate(doObj.getExpiryDate());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public InboundOrderItemDO toDOItem(InboundOrderItem entity) {
        if (entity == null) {
            return null;
        }
        InboundOrderItemDO doObj = new InboundOrderItemDO();
        doObj.setId(entity.getId());
        doObj.setOrderId(entity.getOrderId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setProductName(entity.getProductName());
        doObj.setPlanQuantity(entity.getPlanQuantity());
        doObj.setActualQuantity(entity.getActualQuantity());
        doObj.setLocationId(entity.getLocationId());
        doObj.setStatus(entity.getStatus());
        doObj.setUnitCost(entity.getUnitCost());
        doObj.setBatchNo(entity.getBatchNo());
        doObj.setProductionDate(entity.getProductionDate());
        doObj.setExpiryDate(entity.getExpiryDate());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}