package com.metawebthree.logistics.infrastructure.persistence.converter;

import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.LogisticsOrderDO;
import org.springframework.stereotype.Component;

@Component
public class LogisticsOrderConverter {

    public LogisticsOrder toEntity(LogisticsOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        LogisticsOrder entity = new LogisticsOrder();
        entity.setId(doObj.getId());
        entity.setTrackingNo(doObj.getTrackingNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setCarrierId(doObj.getCarrierId());
        entity.setCarrierName(doObj.getCarrierName());
        entity.setServiceType(doObj.getServiceType());
        entity.setSenderName(doObj.getSenderName());
        entity.setSenderPhone(doObj.getSenderPhone());
        entity.setSenderProvince(doObj.getSenderProvince());
        entity.setSenderCity(doObj.getSenderCity());
        entity.setSenderDistrict(doObj.getSenderDistrict());
        entity.setSenderAddress(doObj.getSenderAddress());
        entity.setReceiverName(doObj.getReceiverName());
        entity.setReceiverPhone(doObj.getReceiverPhone());
        entity.setReceiverProvince(doObj.getReceiverProvince());
        entity.setReceiverCity(doObj.getReceiverCity());
        entity.setReceiverDistrict(doObj.getReceiverDistrict());
        entity.setReceiverAddress(doObj.getReceiverAddress());
        entity.setWeight(doObj.getWeight());
        entity.setVolume(doObj.getVolume());
        entity.setFreight(doObj.getFreight());
        entity.setStatus(doObj.getStatus());
        entity.setPickedUpAt(doObj.getPickedUpAt());
        entity.setInTransitAt(doObj.getInTransitAt());
        entity.setOutForDeliveryAt(doObj.getOutForDeliveryAt());
        entity.setDeliveredAt(doObj.getDeliveredAt());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public LogisticsOrderDO toDO(LogisticsOrder entity) {
        if (entity == null) {
            return null;
        }
        LogisticsOrderDO doObj = new LogisticsOrderDO();
        doObj.setId(entity.getId());
        doObj.setTrackingNo(entity.getTrackingNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setCarrierId(entity.getCarrierId());
        doObj.setCarrierName(entity.getCarrierName());
        doObj.setServiceType(entity.getServiceType());
        doObj.setSenderName(entity.getSenderName());
        doObj.setSenderPhone(entity.getSenderPhone());
        doObj.setSenderProvince(entity.getSenderProvince());
        doObj.setSenderCity(entity.getSenderCity());
        doObj.setSenderDistrict(entity.getSenderDistrict());
        doObj.setSenderAddress(entity.getSenderAddress());
        doObj.setReceiverName(entity.getReceiverName());
        doObj.setReceiverPhone(entity.getReceiverPhone());
        doObj.setReceiverProvince(entity.getReceiverProvince());
        doObj.setReceiverCity(entity.getReceiverCity());
        doObj.setReceiverDistrict(entity.getReceiverDistrict());
        doObj.setReceiverAddress(entity.getReceiverAddress());
        doObj.setWeight(entity.getWeight());
        doObj.setVolume(entity.getVolume());
        doObj.setFreight(entity.getFreight());
        doObj.setStatus(entity.getStatus());
        doObj.setPickedUpAt(entity.getPickedUpAt());
        doObj.setInTransitAt(entity.getInTransitAt());
        doObj.setOutForDeliveryAt(entity.getOutForDeliveryAt());
        doObj.setDeliveredAt(entity.getDeliveredAt());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}