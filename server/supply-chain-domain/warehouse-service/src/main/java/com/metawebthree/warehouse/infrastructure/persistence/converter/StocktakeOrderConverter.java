package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.StocktakeOrder;
import com.metawebthree.warehouse.domain.entity.StocktakeOrderItem;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.StocktakeOrderDO;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.StocktakeOrderItemDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class StocktakeOrderConverter {

    public StocktakeOrder toEntity(StocktakeOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        StocktakeOrder entity = new StocktakeOrder();
        entity.setId(doObj.getId());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setType(doObj.getType());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setWarehouseName(doObj.getWarehouseName());
        entity.setLocationId(doObj.getLocationId());
        entity.setLocationName(doObj.getLocationName());
        entity.setStatus(doObj.getStatus());
        entity.setOperator(doObj.getOperator());
        entity.setPlannedDate(doObj.getPlannedDate());
        entity.setStartDate(doObj.getStartDate());
        entity.setEndDate(doObj.getEndDate());
        entity.setTotalSkuCount(doObj.getTotalSkuCount());
        entity.setCheckedSkuCount(doObj.getCheckedSkuCount());
        entity.setDiscrepancyCount(doObj.getDiscrepancyCount());
        entity.setTotalDiscrepancyAmount(doObj.getTotalDiscrepancyAmount());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public StocktakeOrderDO toDO(StocktakeOrder entity) {
        if (entity == null) {
            return null;
        }
        StocktakeOrderDO doObj = new StocktakeOrderDO();
        doObj.setId(entity.getId());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setType(entity.getType());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setWarehouseName(entity.getWarehouseName());
        doObj.setLocationId(entity.getLocationId());
        doObj.setLocationName(entity.getLocationName());
        doObj.setStatus(entity.getStatus());
        doObj.setOperator(entity.getOperator());
        doObj.setPlannedDate(entity.getPlannedDate());
        doObj.setStartDate(entity.getStartDate());
        doObj.setEndDate(entity.getEndDate());
        doObj.setTotalSkuCount(entity.getTotalSkuCount());
        doObj.setCheckedSkuCount(entity.getCheckedSkuCount());
        doObj.setDiscrepancyCount(entity.getDiscrepancyCount());
        doObj.setTotalDiscrepancyAmount(entity.getTotalDiscrepancyAmount());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    public StocktakeOrderItem toEntityItem(StocktakeOrderItemDO doObj) {
        if (doObj == null) {
            return null;
        }
        StocktakeOrderItem entity = new StocktakeOrderItem();
        entity.setId(doObj.getId());
        entity.setStocktakeOrderId(doObj.getStocktakeOrderId());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setSkuName(doObj.getSkuName());
        entity.setUnit(doObj.getUnit());
        entity.setSystemQuantity(doObj.getSystemQuantity());
        entity.setCountedQuantity(doObj.getCountedQuantity());
        entity.setDiscrepancyQuantity(doObj.getDiscrepancyQuantity());
        entity.setDiscrepancyAmount(doObj.getDiscrepancyAmount());
        entity.setDiscrepancyReason(doObj.getDiscrepancyReason());
        entity.setStatus(doObj.getStatus());
        entity.setCounter(doObj.getCounter());
        entity.setCountedAt(doObj.getCountedAt());
        entity.setChecker(doObj.getChecker());
        entity.setCheckedAt(doObj.getCheckedAt());
        entity.setAdjuster(doObj.getAdjuster());
        entity.setAdjustedAt(doObj.getAdjustedAt());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public StocktakeOrderItemDO toDOItem(StocktakeOrderItem entity) {
        if (entity == null) {
            return null;
        }
        StocktakeOrderItemDO doObj = new StocktakeOrderItemDO();
        doObj.setId(entity.getId());
        doObj.setStocktakeOrderId(entity.getStocktakeOrderId());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setSkuName(entity.getSkuName());
        doObj.setUnit(entity.getUnit());
        doObj.setSystemQuantity(entity.getSystemQuantity());
        doObj.setCountedQuantity(entity.getCountedQuantity());
        doObj.setDiscrepancyQuantity(entity.getDiscrepancyQuantity());
        doObj.setDiscrepancyAmount(entity.getDiscrepancyAmount());
        doObj.setDiscrepancyReason(entity.getDiscrepancyReason());
        doObj.setStatus(entity.getStatus());
        doObj.setCounter(entity.getCounter());
        doObj.setCountedAt(entity.getCountedAt());
        doObj.setChecker(entity.getChecker());
        doObj.setCheckedAt(entity.getCheckedAt());
        doObj.setAdjuster(entity.getAdjuster());
        doObj.setAdjustedAt(entity.getAdjustedAt());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    public List<StocktakeOrder> toEntityList(List<StocktakeOrderDO> doList) {
        if (doList == null) {
            return null;
        }
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }

    public List<StocktakeOrderDO> toDOList(List<StocktakeOrder> entityList) {
        if (entityList == null) {
            return null;
        }
        return entityList.stream().map(this::toDO).collect(Collectors.toList());
    }

    public List<StocktakeOrderItem> toEntityItemList(List<StocktakeOrderItemDO> doList) {
        if (doList == null) {
            return null;
        }
        return doList.stream().map(this::toEntityItem).collect(Collectors.toList());
    }

    public List<StocktakeOrderItemDO> toDOItemList(List<StocktakeOrderItem> entityList) {
        if (entityList == null) {
            return null;
        }
        return entityList.stream().map(this::toDOItem).collect(Collectors.toList());
    }
}
